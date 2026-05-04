"""
Fashion-MNIST Optimizer Comparison (Pure NumPy)
==============================================
Reproduces the optimizer comparison style from:
'"The Marginal Value of Adaptive Gradient Methods in Machine Learning"'
(Ashia C. Wilson et al., 2017)

Experiment goals:
1. Train a 2-layer MLP on Fashion-MNIST.
2. Compare Adam vs SGD+Momentum.
3. Tune learning rates independently via grid search.
4. Select best validation model per optimizer.
5. Evaluate on test set.
6. Repeat across 5 random seeds.
7. Compute 95% confidence intervals.
8. Save training/validation plots.

Dataset format (local CSV files expected):
./dataset/fashion-mnist_train.csv
./dataset/fashion-mnist_test.csv

CSV schema:
label,pixel1,pixel2,...,pixel784
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


# ============================================================
# DATA LOADING
# ============================================================
def load_csv_dataset(data_dir: str = "dataset") -> Tuple[np.ndarray, ...]:
    """Load Fashion-MNIST from local CSV files.

    Args:
        data_dir: Directory containing train/test CSV files.

    Returns:
        X_train, y_train, X_test, y_test
    """
    train_path = os.path.join(data_dir, "fashion-mnist_train.csv")
    test_path = os.path.join(data_dir, "fashion-mnist_test.csv")

    train_data = np.loadtxt(train_path, delimiter=",", skiprows=1)
    test_data = np.loadtxt(test_path, delimiter=",", skiprows=1)

    y_train = train_data[:, 0].astype(int)
    X_train = train_data[:, 1:].astype(np.float32) / 255.0

    y_test = test_data[:, 0].astype(int)
    X_test = test_data[:, 1:].astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert class labels into one-hot vectors."""
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy."""
    return np.mean(pred == y)


def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax implementation."""
    x = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy loss."""
    eps = 1e-12
    return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))


def confidence_interval(values: List[float]) -> Tuple[float, float, float]:
    """Compute mean, std, and 95% confidence interval."""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ci = 1.96 * std / np.sqrt(len(values))
    return mean, std, ci


# ============================================================
# MODEL
# ============================================================
class MLP:
    """Two-layer MLP implemented from scratch using NumPy."""

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, seed=42):
        rng = np.random.RandomState(seed)

        # He initialization for ReLU
        self.params = {
            "W1": rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
            "b1": np.zeros((1, hidden_dim)),
            "W2": rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim),
            "b2": np.zeros((1, output_dim)),
        }

    def forward(self, X: np.ndarray):
        """Forward pass."""
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        probs = softmax(z2)
        return probs, (X, z1, a1, probs)

    def backward(self, cache, y: np.ndarray):
        """Manual backpropagation."""
        X, z1, a1, probs = cache
        m = X.shape[0]
        y_onehot = one_hot(y)

        # Output layer gradients
        dz2 = (probs - y_onehot) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)


# ============================================================
# OPTIMIZERS
# ============================================================
class SGD_Momentum:
    """SGD with momentum."""

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        """Update parameters using momentum."""
        for k in params:
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += self.v[k]


class Adam:
    """Adam optimizer."""

    def __init__(self, params, lr=0.001):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        """Update parameters using Adam."""
        self.t += 1
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# TRAINING LOOP
# ============================================================
def iterate_minibatches(X, y, batch_size, rng):
    """Yield shuffled mini-batches."""
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train_model(optimizer_name, lr, seed, X_train, y_train, X_val, y_val,
                X_test, y_test, epochs=600, batch_size=1024):
    """Train one model instance and return best validation model."""
    rng = np.random.RandomState(seed)
    model = MLP(seed=seed)

    optimizer = SGD_Momentum(model.params, lr) if optimizer_name == "sgd" else Adam(model.params, lr)

    history = {
    "train_loss": [],
    "train_acc": [],
    "train_error": [],
    "val_acc": [],
    "test_loss": [],
    "test_acc": [],
    "test_error": [],
    }
    best_val_acc = -1
    best_params = None

    for epoch in range(epochs):
        # Mini-batch training
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, rng):
            probs, cache = model.forward(xb)
            grads = model.backward(cache, yb)
            optimizer.step(model.params, grads)

        # Training metrics
        train_probs, _ = model.forward(X_train)
        train_preds = np.argmax(train_probs, axis=1)

        train_loss = cross_entropy(train_probs, y_train)
        train_acc = accuracy(train_preds, y_train)
        train_error = 1.0 - train_acc

        # Validation accuracy
        val_acc = accuracy(model.predict(X_val), y_val)

        # Test metrics
        test_probs, _ = model.forward(X_test)
        test_preds = np.argmax(test_probs, axis=1)

        test_loss = cross_entropy(test_probs, y_test)
        test_acc = accuracy(test_preds, y_test)
        test_error = 1.0 - test_acc

        # Store history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_error"].append(train_error)

        history["val_acc"].append(val_acc)

        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_error"].append(test_error)

        # Save best validation model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

    model.params = best_params
    return model, best_val_acc, history


def run_grid_search(optimizer_name, lr_grid, seed,
                    X_train, y_train, X_val, y_val,
                    X_test, y_test):
    """Run learning-rate grid search for a given optimizer."""
    best_model = None
    best_lr = None
    best_val_acc = -1
    best_history = None
    all_histories = {}  # lr -> history, for per-LR plots

    for lr in lr_grid:
        print(f"[{optimizer_name}] Seed={seed} LR={lr}")

        model, val_acc, history = train_model(
            optimizer_name, lr, seed,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
        )

        all_histories[lr] = history

        # Display final metrics for this LR variation
        final_train_loss = history["train_loss"][-1]
        final_test_loss = history["test_loss"][-1]
        final_val_acc = history["val_acc"][-1]
        final_train_acc = history["train_acc"][-1]
        final_train_error = history["train_error"][-1]
        final_test_acc = history["test_acc"][-1]
        final_test_error = history["test_error"][-1]

        print(
            f"[{optimizer_name.upper()} | Seed={seed} | LR={lr}] "
            f"Train Loss: {final_train_loss:.4f} | "
            f"Train Acc: {final_train_acc:.4f} | "
            f"Train Err: {final_train_error:.4f} | "
            f"Test Loss: {final_test_loss:.4f} | "
            f"Test Acc: {final_test_acc:.4f} | "
            f"Test Err: {final_test_error:.4f} | "
            f"Val Acc: {final_val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Deep copy model so it preserves this LR's best state
            best_model = MLP(seed=seed)
            best_model.params = {
                k: v.copy() for k, v in model.params.items()
            }

            best_lr = lr
            best_history = history

    return best_model, best_lr, best_val_acc, best_history, all_histories


# ============================================================
# PLOTTING
# ============================================================
def lr_folder(base_dir: str, optimizer_name: str, lr: float) -> str:
    """Return (and create) a per-LR sub-folder."""
    lr_str = f"lr_{lr:.0e}".replace("+", "").replace("-0", "-")
    folder = os.path.join(base_dir, optimizer_name, lr_str)
    os.makedirs(folder, exist_ok=True)
    return folder


def plot_combined(adam_history, sgd_history, seed: int,
                  adam_lr: float, sgd_lr: float,
                  base_dir: str = "numpy_plots"):
    """Save combined Adam vs SGD plots (loss, accuracy, test loss) for a seed."""
    out_dir = os.path.join(base_dir, f"seed_{seed}_combined")
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(adam_history["train_loss"]) + 1)

    # --- Training loss ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, adam_history["train_loss"], label=f"Adam (lr={adam_lr})")
    ax.plot(epochs, sgd_history["train_loss"],  label=f"SGD  (lr={sgd_lr})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss — Seed {seed}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "train_loss.png"))
    plt.close(fig)

    # --- Validation accuracy ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, adam_history["val_acc"], label=f"Adam (lr={adam_lr})")
    ax.plot(epochs, sgd_history["val_acc"],  label=f"SGD  (lr={sgd_lr})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Validation Accuracy — Seed {seed}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "val_accuracy.png"))
    plt.close(fig)

    # --- Test loss ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, adam_history["test_loss"], label=f"Adam (lr={adam_lr})")
    ax.plot(epochs, sgd_history["test_loss"],  label=f"SGD  (lr={sgd_lr})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Test Loss — Seed {seed}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "test_loss.png"))
    plt.close(fig)

    # --- Training Accuracy ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, adam_history["train_acc"], label=f"Adam (lr={adam_lr})")
    ax.plot(epochs, sgd_history["train_acc"], label=f"SGD (lr={sgd_lr})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Training Accuracy — Seed {seed}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "train_accuracy.png"))
    plt.close(fig)


    # --- Training Error ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, adam_history["train_error"], label=f"Adam (lr={adam_lr})")
    ax.plot(epochs, sgd_history["train_error"], label=f"SGD (lr={sgd_lr})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_title(f"Training Error — Seed {seed}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "train_error.png"))
    plt.close(fig)


    # --- Test Accuracy ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, adam_history["test_acc"], label=f"Adam (lr={adam_lr})")
    ax.plot(epochs, sgd_history["test_acc"], label=f"SGD (lr={sgd_lr})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Test Accuracy — Seed {seed}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "test_accuracy.png"))
    plt.close(fig)


    # --- Test Error ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, adam_history["test_error"], label=f"Adam (lr={adam_lr})")
    ax.plot(epochs, sgd_history["test_error"], label=f"SGD (lr={sgd_lr})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_title(f"Test Error — Seed {seed}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "test_error.png"))
    plt.close(fig)


def plot_per_lr(all_adam_histories: Dict, all_sgd_histories: Dict,
                seed: int, base_dir: str = "numpy_plots"):
    """Save per-LR plots with Adam and SGD overlaid where LRs match, or solo otherwise."""
    adam_lrs = set(all_adam_histories.keys())
    sgd_lrs  = set(all_sgd_histories.keys())
    all_lrs  = adam_lrs | sgd_lrs

    for lr in all_lrs:
        has_adam = lr in adam_lrs
        has_sgd  = lr in sgd_lrs

        # Determine output folder (prefer optimizer-specific when only one exists)
        if has_adam and has_sgd:
            folder = os.path.join(base_dir, "both", f"lr_{lr:.0e}", f"seed_{seed}")
        elif has_adam:
            folder = lr_folder(base_dir, "adam", lr)
            folder = os.path.join(folder, f"seed_{seed}")
        else:
            folder = lr_folder(base_dir, "sgd", lr)
            folder = os.path.join(folder, f"seed_{seed}")
        os.makedirs(folder, exist_ok=True)

        epochs_adam = range(1, len(all_adam_histories[lr]["train_loss"]) + 1) if has_adam else None
        epochs_sgd  = range(1, len(all_sgd_histories[lr]["train_loss"])  + 1) if has_sgd  else None

        for metric, ylabel, title_prefix in [
            ("train_loss", "Loss", "Training Loss"),
            ("train_acc", "Accuracy", "Training Accuracy"),
            ("train_error", "Error", "Training Error"),
            ("val_acc", "Accuracy", "Validation Accuracy"),
            ("test_loss", "Loss", "Test Loss"),
            ("test_acc", "Accuracy", "Test Accuracy"),
            ("test_error", "Error", "Test Error"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 6))
            if has_adam:
                ax.plot(epochs_adam, all_adam_histories[lr][metric], label=f"Adam")
            if has_sgd:
                ax.plot(epochs_sgd,  all_sgd_histories[lr][metric],  label=f"SGD")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title_prefix} — LR={lr:.0e}, Seed {seed}")
            if has_adam or has_sgd:
                ax.legend()
            fig.savefig(os.path.join(folder, f"{metric}.png"))
            plt.close(fig)

def plot_confidence_bands(
    adam_histories: List[Dict],
    sgd_histories: List[Dict],
    base_dir: str = "numpy_plots"
):
    """
    Plot mean training/test/validation curves across all seeds
    with 95% confidence bands.

    Each curve uses the best learning-rate history chosen for that seed.
    """
    out_dir = os.path.join(base_dir, "confidence_bands")
    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        ("train_loss", "Loss", "Training Loss"),
        ("train_acc", "Accuracy", "Training Accuracy"),
        ("train_error", "Error", "Training Error"),
        ("test_loss", "Loss", "Test Loss"),
        ("test_acc", "Accuracy", "Test Accuracy"),
        ("test_error", "Error", "Test Error"),
        ("val_acc", "Accuracy", "Validation Accuracy"),
    ]

    for metric, ylabel, title in metrics:
        # Shape: (num_seeds, num_epochs)
        adam_matrix = np.array([h[metric] for h in adam_histories])
        sgd_matrix = np.array([h[metric] for h in sgd_histories])

        epochs = np.arange(1, adam_matrix.shape[1] + 1)

        # Means across seeds
        adam_mean = np.mean(adam_matrix, axis=0)
        sgd_mean = np.mean(sgd_matrix, axis=0)

        # Standard deviations across seeds
        adam_std = np.std(adam_matrix, axis=0, ddof=1)
        sgd_std = np.std(sgd_matrix, axis=0, ddof=1)

        # 95% confidence intervals
        adam_ci = 1.96 * adam_std / np.sqrt(len(adam_histories))
        sgd_ci = 1.96 * sgd_std / np.sqrt(len(sgd_histories))

        fig, ax = plt.subplots(figsize=(8, 6))

        # Mean curves
        ax.plot(epochs, adam_mean, label="Adam")
        ax.plot(epochs, sgd_mean, label="SGD")

        # Confidence bands
        ax.fill_between(
            epochs,
            adam_mean - adam_ci,
            adam_mean + adam_ci,
            alpha=0.25
        )

        ax.fill_between(
            epochs,
            sgd_mean - sgd_ci,
            sgd_mean + sgd_ci,
            alpha=0.25
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (Mean ± 95% CI Across Seeds)")
        ax.legend()

        fig.savefig(
            os.path.join(
                out_dir,
                f"{metric}_confidence_band.png"
            )
        )
        plt.close(fig)

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def main():
    """Run full optimizer comparison experiment."""
    X_train_full, y_train_full, X_test, y_test = load_csv_dataset()

    # Fixed train/validation split
    X_train = X_train_full[:50000]
    y_train = y_train_full[:50000]
    X_val = X_train_full[50000:60000]
    y_val = y_train_full[50000:60000]

    seeds = [0, 1, 2, 3, 4]

    # Independent learning-rate grids
    adam_grid = [1e-4, 3e-4, 1e-3, 3e-3]
    sgd_grid = [1e-3, 1e-2, 5e-2, 1e-1]

    # Final test accuracies (best model per seed)
    adam_results = []
    sgd_results = []

    # Best training histories per seed
    # Used for confidence-band plots across seeds
    adam_seed_histories = []
    sgd_seed_histories = []

    for seed in seeds:
        print("=" * 60)
        print(f"Running Seed {seed}")

        # Adam search
        adam_model, adam_lr, _, adam_history, all_adam_histories = run_grid_search(
            "adam", adam_grid, seed,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
        )

        # SGD search
        sgd_model, sgd_lr, _, sgd_history, all_sgd_histories = run_grid_search(
            "sgd", sgd_grid, seed,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
        )

        # Test evaluation
        adam_test_acc = accuracy(adam_model.predict(X_test), y_test)
        sgd_test_acc = accuracy(sgd_model.predict(X_test), y_test)

        # Store final test accuracies
        adam_results.append(adam_test_acc)
        sgd_results.append(sgd_test_acc)

        # Store best-history trajectories for cross-seed aggregation
        adam_seed_histories.append(adam_history)
        sgd_seed_histories.append(sgd_history)

        # Combined plots (best LR for each optimizer side by side)
        plot_combined(adam_history, sgd_history, seed, adam_lr, sgd_lr)

        # Per-LR plots (Adam and SGD overlaid per shared LR; solo otherwise)
        plot_per_lr(all_adam_histories, all_sgd_histories, seed)

        print(f"Adam best LR: {adam_lr} | Test Acc: {adam_test_acc:.4f}")
        print(f"SGD best LR:  {sgd_lr} | Test Acc: {sgd_test_acc:.4f}")

    # Cross-seed confidence-band plots
    plot_confidence_bands(
        adam_seed_histories,
        sgd_seed_histories
    )
    # Aggregate statistics
    adam_mean, adam_std, adam_ci = confidence_interval(adam_results)
    sgd_mean, sgd_std, sgd_ci = confidence_interval(sgd_results)

    gap = sgd_mean - adam_mean

    print("\nFinal Results")
    print("=" * 60)
    print(f"Adam Mean Test Accuracy: {adam_mean:.4f} ± {adam_ci:.4f}")
    print(f"SGD  Mean Test Accuracy: {sgd_mean:.4f} ± {sgd_ci:.4f}")
    print(f"Generalization Gap (SGD - Adam): {gap:.4f}")


if __name__ == "__main__":
    main()