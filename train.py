"""
Main training entry point — managed by Hydra.

Run from the project root (after cd-ing into it in Colab):

    # Default (convnext_small, PCA mode, 3 components)
    python train.py

    # ── Spectral mode ────────────────────────────────────────────────────────
    # PCA mode — change number of PCA components (in_channels auto-updates)
    python train.py data.pca_components=10

    # Conv reducer mode (learnable 150→32→3 conv inside the model)
    python train.py model=convnext_small_conv data.spectral_mode=conv

    # Conv mode with more reducer output channels
    python train.py model=convnext_small_conv data.spectral_mode=conv model.reducer_channels=10

    # ── Model selection ───────────────────────────────────────────────────────
    # PCA mode models:  convnext_tiny | convnext_small | convnext_large | efficientnet_b0
    # Conv mode models: convnext_tiny_conv | convnext_small_conv | convnext_large_conv
    python train.py model=convnext_large

    # ── Loss function ─────────────────────────────────────────────────────────
    python train.py training.loss=smooth_l1
    python train.py training.loss=huber

    # ── Combined overrides ────────────────────────────────────────────────────
    python train.py model=convnext_large data.pca_components=10 training.loss=smooth_l1
    python train.py model=convnext_tiny_conv data.spectral_mode=conv training.epochs=50

    # ── Misc ──────────────────────────────────────────────────────────────────
    python train.py training.optimizer.lr_backbone=1e-4 training.epochs=50
    python train.py wandb.enabled=false
"""

import os
import sys
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

# Allow importing from src/ regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from src.dataset import (
    load_data, load_gt,
    collect_valid_pixels, transform_patches_with_mask,
    calculate_global_stats, NPZDataset,
)
from src.model   import HyperspectralRegressor
from src.train   import train_with_early_stopping
from src.evaluate import (
    SpectralCurveFiltering, BaselineRegressor,
    evaluate_dl_model, evaluate_baseline_regressor,
    calculate_and_print_results,
)


def build_criterion(loss_name: str) -> nn.Module:
    """Return the requested loss function."""
    name = loss_name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss(beta=1.0)
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(
        f"Unknown loss '{loss_name}'. Choose from: mse, smooth_l1, huber"
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    spectral_mode = cfg.data.spectral_mode   # "pca" | "conv"
    assert spectral_mode in ("pca", "conv"), \
        f"data.spectral_mode must be 'pca' or 'conv', got '{spectral_mode}'"

    # ------------------------------------------------------------------ setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Spectral mode: {spectral_mode}")

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    pad_size = tuple(cfg.data.pad_size)

    # ------------------------------------------------------------------ data
    print("\n[1/6] Loading raw data …")
    X_all = load_data(cfg.paths.train_data)
    y_all = load_gt(cfg.paths.gt_path)

    # ---- split ----
    indices = np.arange(len(X_all))
    X_tmp, X_test, y_tmp, y_test, idx_tmp, idx_test = train_test_split(
        X_all, y_all, indices,
        test_size=cfg.data.split.test_size,
        random_state=cfg.data.split.random_state,
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_tmp, y_tmp, idx_tmp,
        test_size=cfg.data.split.val_size,
        random_state=cfg.data.split.random_state,
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ------------------------------------------------------------------ spectral reduction
    if spectral_mode == "pca":
        print(f"\n[2/6] Fitting PCA ({cfg.data.pca_components} components) …")
        pixels_train = collect_valid_pixels(X_train, max_pixels=cfg.data.max_pixels_for_pca)
        print(f"  Collected {pixels_train.shape[0]:,} valid pixels")

        scaler_x = StandardScaler()
        pixels_scaled = scaler_x.fit_transform(pixels_train)

        pca = PCA(n_components=cfg.data.pca_components)
        pca.fit(pixels_scaled)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

        joblib.dump(scaler_x, cfg.paths.scaler_x_save)
        joblib.dump(pca,      cfg.paths.pca_save)

        X_train_ds = transform_patches_with_mask(X_train, scaler_x, pca)
        X_val_ds   = transform_patches_with_mask(X_val,   scaler_x, pca)
        X_test_ds  = transform_patches_with_mask(X_test,  scaler_x, pca)

    else:  # conv mode — raw 150-band data fed directly to the model
        print("\n[2/6] Conv reducer mode — skipping PCA, using raw 150-band data …")
        raw_ch = X_all[0][0].shape[0]
        print(f"  Raw spectral channels: {raw_ch}")
        X_train_ds, X_val_ds, X_test_ds = X_train, X_val, X_test

    # ------------------------------------------------------------------ label scaler
    print("\n[3/6] Scaling labels …")
    scaler_y = StandardScaler()
    y_train_sc = scaler_y.fit_transform(y_train)
    y_val_sc   = scaler_y.transform(y_val)
    y_test_sc  = scaler_y.transform(y_test)
    joblib.dump(scaler_y, cfg.paths.scaler_y_save)

    # ------------------------------------------------------------------ global stats
    global_means, global_stds = calculate_global_stats(X_train_ds, pad_size=pad_size)
    torch.save(
        {"means": global_means, "stds": global_stds},
        cfg.paths.global_stats_save,
    )
    print(f"  Global stats saved → {cfg.paths.global_stats_save}")

    # ------------------------------------------------------------------ datasets / loaders
    print("\n[4/6] Building datasets …")
    aug_cfg = cfg.data.augment

    train_ds = NPZDataset(X_train_ds, y_train_sc, augment=True,
                          size=pad_size, global_means=global_means,
                          global_stds=global_stds, aug_cfg=aug_cfg)
    val_ds   = NPZDataset(X_val_ds,   y_val_sc,   augment=False,
                          size=pad_size, global_means=global_means,
                          global_stds=global_stds)
    test_ds  = NPZDataset(X_test_ds,  y_test_sc,  augment=False,
                          size=pad_size, global_means=global_means,
                          global_stds=global_stds)

    loader_kw = dict(batch_size=cfg.data.batch_size,
                     num_workers=cfg.data.num_workers,
                     pin_memory=cfg.data.pin_memory)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    # ------------------------------------------------------------------ model
    print("\n[5/6] Building model …")

    # raw_in_channels matters only in conv mode (detected from the data)
    raw_in_channels = X_all[0][0].shape[0] if spectral_mode == "conv" else 150

    model = HyperspectralRegressor(
        in_channels          = cfg.model.in_channels,
        n_outputs            = 4,
        backbone_name        = cfg.model.backbone_name,
        pretrained           = cfg.model.pretrained,
        dropout              = cfg.model.dropout,
        use_spectral_reducer = cfg.model.use_spectral_reducer,
        raw_in_channels      = raw_in_channels,
        reducer_mid_channels = getattr(cfg.model, "reducer_mid_channels", 32),
    ).to(device)

    # ---- optimizer (3 groups in conv mode, 2 in PCA mode) ----
    opt_cfg = cfg.training.optimizer
    if cfg.model.use_spectral_reducer:
        param_groups = [
            {"params": model.spectral_reducer.parameters(), "lr": opt_cfg.lr_reducer},
            {"params": model.backbone.parameters(),         "lr": opt_cfg.lr_backbone},
            {"params": model.regressor.parameters(),        "lr": opt_cfg.lr_head},
        ]
    else:
        param_groups = [
            {"params": model.backbone.parameters(),  "lr": opt_cfg.lr_backbone},
            {"params": model.regressor.parameters(), "lr": opt_cfg.lr_head},
        ]
    optimizer = optim.AdamW(param_groups, weight_decay=opt_cfg.weight_decay)

    # ---- loss function ----
    criterion = build_criterion(cfg.training.loss)
    print(f"  Loss: {cfg.training.loss}  |  Optimizer groups: {len(param_groups)}")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.scheduler.factor,
        patience=cfg.training.scheduler.patience,
        min_lr=cfg.training.scheduler.min_lr,
    )

    # ------------------------------------------------------------------ W&B
    run = None
    if cfg.wandb.enabled:
        # API key priority:
        #   1. cfg.wandb.api_key  (set via CLI: wandb.api_key=<key>)
        #   2. WANDB_API_KEY env var  (set in Colab Secrets cell)
        #   3. Saved token (~/.netrc) — already logged-in machines
        #   4. Interactive prompt     — fallback in interactive sessions
        api_key = cfg.wandb.api_key or os.environ.get("WANDB_API_KEY") or None
        wandb.login(key=api_key)

        run = wandb.init(
            project = cfg.wandb.project,
            name    = cfg.wandb.name,
            notes   = cfg.wandb.notes,
            tags    = list(cfg.wandb.tags),
            group   = cfg.wandb.group or None,
            config  = OmegaConf.to_container(cfg, resolve=True),
        )

    # ------------------------------------------------------------------ train
    print("\n[6/6] Training …")
    train_with_early_stopping(
        model         = model,
        train_loader  = train_loader,
        val_loader    = val_loader,
        optimizer     = optimizer,
        scheduler     = scheduler,
        criterion     = criterion,
        device        = device,
        epochs        = cfg.training.epochs,
        warmup_epochs = cfg.training.warmup_epochs,
        patience      = cfg.training.patience,
        save_path     = cfg.paths.model_save,
        max_grad_norm = cfg.training.max_grad_norm,
        wandb_run     = run,
    )

    # ------------------------------------------------------------------ evaluate
    print("\n[Eval] Evaluating on local test set …")

    # DL model
    dl_preds, y_test_true, dl_mse = evaluate_dl_model(model, test_loader, scaler_y, device)

    # Baseline (mean spectrum → mean label)  — always uses raw 150-band data
    filtering = SpectralCurveFiltering()
    X_test_raw      = [X_all[i][0] for i in idx_test]
    X_test_filtered = np.array([filtering(c.cpu().numpy()) for c in X_test_raw])

    combined_idx    = idx_train.tolist() + idx_val.tolist()
    X_tv_raw        = [X_all[i][0] for i in combined_idx]
    X_tv_filtered   = np.array([filtering(c.cpu().numpy()) for c in X_tv_raw])
    y_tv            = y_all[combined_idx]

    baseline = BaselineRegressor().fit(X_tv_filtered, y_tv)
    baseline_preds, baseline_mse = evaluate_baseline_regressor(
        baseline, X_test_filtered, y_test_true
    )

    score = calculate_and_print_results(dl_mse, baseline_mse, y_test_true, dl_preds, baseline_preds)

    if run is not None:
        run.summary["challenge_score"] = score
        wandb.finish()


if __name__ == "__main__":
    main()
