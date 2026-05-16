"""
Main training entry point — managed by Hydra.

Run from the project root (after cd-ing into it in Colab):

    # Default config (convnext_small, pca_components=3)
    python train.py

    # Change model
    python train.py model=convnext_tiny

    # Change PCA components (model.in_channels updates automatically)
    python train.py data.pca_components=10

    # Change model + PCA together
    python train.py model=efficientnet_b0 data.pca_components=5

    # Override any parameter
    python train.py training.optimizer.lr_backbone=1e-4 training.epochs=50

    # Disable W&B
    python train.py wandb.enabled=false
"""

import os
import sys
import joblib
import numpy as np
import torch
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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------ setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    # ------------------------------------------------------------------ PCA
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

    X_train_pca = transform_patches_with_mask(X_train, scaler_x, pca)
    X_val_pca   = transform_patches_with_mask(X_val,   scaler_x, pca)
    X_test_pca  = transform_patches_with_mask(X_test,  scaler_x, pca)

    # ------------------------------------------------------------------ label scaler
    print("\n[3/6] Scaling labels …")
    scaler_y = StandardScaler()
    y_train_sc = scaler_y.fit_transform(y_train)
    y_val_sc   = scaler_y.transform(y_val)
    y_test_sc  = scaler_y.transform(y_test)
    joblib.dump(scaler_y, cfg.paths.scaler_y_save)

    # ------------------------------------------------------------------ global stats
    global_means, global_stds = calculate_global_stats(X_train_pca, pad_size=pad_size)

    # ------------------------------------------------------------------ datasets / loaders
    print("\n[4/6] Building datasets …")
    aug_cfg = cfg.data.augment

    train_ds = NPZDataset(X_train_pca, y_train_sc, augment=True,
                          size=pad_size, global_means=global_means,
                          global_stds=global_stds, aug_cfg=aug_cfg)
    val_ds   = NPZDataset(X_val_pca,   y_val_sc,   augment=False,
                          size=pad_size, global_means=global_means,
                          global_stds=global_stds)
    test_ds  = NPZDataset(X_test_pca,  y_test_sc,  augment=False,
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
    model = HyperspectralRegressor(
        in_channels   = cfg.model.in_channels,
        n_outputs     = 4,
        backbone_name = cfg.model.backbone_name,
        pretrained    = cfg.model.pretrained,
        dropout       = cfg.model.dropout,
    ).to(device)

    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(),  "lr": cfg.training.optimizer.lr_backbone},
        {"params": model.regressor.parameters(), "lr": cfg.training.optimizer.lr_head},
    ], weight_decay=cfg.training.optimizer.weight_decay)

    criterion = torch.nn.MSELoss()

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
            name    = cfg.wandb.name,          # e.g. "convnext_tiny_in22k_pca10"
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

    # Baseline (mean spectrum → mean label)
    filtering = SpectralCurveFiltering()
    X_test_raw = [X_all[i][0] for i in idx_test]
    X_test_filtered = np.array([filtering(c.cpu().numpy()) for c in X_test_raw])

    combined_idx = idx_train.tolist() + idx_val.tolist()
    X_tv_raw = [X_all[i][0] for i in combined_idx]
    X_tv_filtered = np.array([filtering(c.cpu().numpy()) for c in X_tv_raw])
    y_tv = y_all[combined_idx]

    baseline = BaselineRegressor().fit(X_tv_filtered, y_tv)
    baseline_preds, baseline_mse = evaluate_baseline_regressor(baseline, X_test_filtered, y_test_true)

    score = calculate_and_print_results(dl_mse, baseline_mse, y_test_true, dl_preds, baseline_preds)

    if run is not None:
        run.summary["challenge_score"] = score
        wandb.finish()


if __name__ == "__main__":
    main()
