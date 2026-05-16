import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Spectral curve filtering (used to prepare features for the baseline)
# ---------------------------------------------------------------------------

class SpectralCurveFiltering:
    """Compute the mean spectrum over all pixels per patch.
    Input : (C, H, W) numpy array
    Output: (C,) mean spectral signature
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x.mean(axis=(1, 2))


# ---------------------------------------------------------------------------
# Baseline regressor
# ---------------------------------------------------------------------------

class BaselineRegressor:
    """Always predicts the training-set mean — used as the competition baseline."""

    def __init__(self):
        self.mean = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.mean         = np.mean(y_train, axis=0)
        self.n_outputs    = y_train.shape[1]
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.full((len(X_test), self.n_outputs), self.mean)


# ---------------------------------------------------------------------------
# Deep-learning model evaluation
# ---------------------------------------------------------------------------

def evaluate_dl_model(model, test_loader, scaler_y, device):
    """Run inference on the test loader and return unscaled predictions + MSE."""
    model.eval()
    preds_scaled, true_scaled = [], []

    with torch.no_grad():
        for X_batch, nv_batch, y_batch in test_loader:
            X_batch  = X_batch.to(device)
            nv_batch = nv_batch.to(device)
            preds_scaled.append(model(X_batch, nv_batch).cpu().numpy())
            true_scaled.append(y_batch.cpu().numpy())

    y_pred = scaler_y.inverse_transform(np.vstack(preds_scaled))
    y_true = scaler_y.inverse_transform(np.vstack(true_scaled))
    mse    = np.mean((y_true - y_pred) ** 2, axis=0)
    return y_pred, y_true, mse


def evaluate_baseline_regressor(baseline, X_test_filtered, y_test_true):
    """Evaluate the baseline regressor and return predictions + MSE."""
    preds = baseline.predict(X_test_filtered)
    mse   = np.mean((y_test_true - preds) ** 2, axis=0)
    return preds, mse


def calculate_and_print_results(model_mse, baseline_mse, y_true, y_pred, baseline_preds):
    """Print per-target MSE comparison and challenge score, then plot."""
    score        = np.mean(model_mse / baseline_mse)
    target_names = ["P", "K", "Mg", "pH"]

    print("\nPer-target comparison:")
    for i, name in enumerate(target_names):
        norm = model_mse[i] / baseline_mse[i]
        print(
            f"  {name}: Model MSE={model_mse[i]:.4f}  "
            f"Baseline MSE={baseline_mse[i]:.4f}  "
            f"Normalized={norm:.4f}"
        )
    print(f"\nChallenge score (lower is better): {score:.4f}")

    plt.figure(figsize=(15, 5))
    for i, name in enumerate(target_names):
        plt.subplot(1, 4, i + 1)
        plt.scatter(y_true[:, i], y_pred[:, i],         alpha=0.7, label="DL Model")
        plt.scatter(y_true[:, i], baseline_preds[:, i], alpha=0.7, label="Baseline", marker="x")
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([lo, hi], [lo, hi], "r--")
        plt.title(name)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    return score
