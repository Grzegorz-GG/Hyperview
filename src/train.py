import torch
import torch.nn as nn
from tqdm import tqdm


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs: int        = 90,
    warmup_epochs: int = 10,
    patience: int      = 13,
    save_path: str     = "best_model.pth",
    max_grad_norm: float = 1.0,
    wandb_run          = None,
):
    """Train with linear LR warm-up, ReduceLROnPlateau, gradient clipping,
    and early stopping. Saves the best checkpoint by validation loss.

    Returns the model loaded with the best weights.
    """
    initial_lrs    = [pg["lr"] for pg in optimizer.param_groups]
    best_val_loss  = float("inf")
    early_stop_ctr = 0

    for epoch in range(epochs):

        # ---- Linear warm-up ----
        if epoch < warmup_epochs:
            wf = (epoch + 1) / float(warmup_epochs)
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = initial_lrs[i] * wf

        # ---- Train ----
        model.train()
        train_sum, n_train = 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for xb, nv_b, yb in loop:
            xb, nv_b, yb = xb.to(device), nv_b.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, nv_b), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_sum += loss.item() * xb.size(0)
            n_train   += xb.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = train_sum / max(1, n_train)

        # ---- Validate ----
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for xv, nv_v, yv in val_loader:
                xv, nv_v, yv = xv.to(device), nv_v.to(device), yv.to(device)
                val_sum += criterion(model(xv, nv_v), yv).item() * xv.size(0)
                n_val   += xv.size(0)
        val_loss = val_sum / max(1, n_val)

        # ---- Scheduler (skip during warm-up) ----
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

        # ---- Logging ----
        # param group order: [reducer, backbone, head] (conv) or [backbone, head] (pca)
        lrs      = [pg["lr"] for pg in optimizer.param_groups]
        lr_names = (["lr_reducer", "lr_backbone", "lr_head"] if len(lrs) == 3
                    else ["lr_backbone", "lr_head"])
        lr_str   = "  ".join(f"{n}: {v:.2e}" for n, v in zip(lr_names, lrs))
        print(
            f"Epoch {epoch+1:3d}/{epochs}  "
            f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  {lr_str}"
        )
        if wandb_run is not None:
            log_dict = {
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "epoch":      epoch + 1,
            }
            log_dict.update(dict(zip(lr_names, lrs)))
            wandb_run.log(log_dict)

        # ---- Early stopping & checkpoint ----
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            early_stop_ctr = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            early_stop_ctr += 1
            if early_stop_ctr >= patience:
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(best val loss: {best_val_loss:.4f})"
                )
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Best model loaded (val loss: {best_val_loss:.4f})")
    return model
