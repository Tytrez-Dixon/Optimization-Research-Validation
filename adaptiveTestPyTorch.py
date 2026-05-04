"""
Fashion-MNIST Optimizer Comparison + Ablation Study (PyTorch)
=============================================================
Reproduces the optimizer comparison style from:
"The Marginal Value of Adaptive Gradient Methods in Machine Learning"
(Ashia C. Wilson et al., 2017)

Experiment goals:
1. Train a 2-layer MLP on Fashion-MNIST.
2. Compare Adam vs SGD+Momentum.
3. Tune learning rates independently via grid search.
4. Select best validation model per optimizer.
5. Evaluate on test set.
6. Repeat across 5 random seeds.
7. Compute 95% confidence intervals.
8. Run optimizer and MLP ablation studies.
9. Save per-LR, combined, confidence-band, and ablation plots.

Dataset format (local CSV files expected):
./dataset/fashion-mnist_train.csv
./dataset/fashion-mnist_test.csv

CSV schema: label,pixel1,...,pixel784

Output structure:
torch_plots/
  lr_<value>/seed_<N>/           Per-LR training curves (Adam + SGD)
  seed_<N>_combined/             Best-LR combined curves per seed
  confidence_bands/              Mean ± 95% CI across seeds
  ablation/
    optimizer/
      <ablation_name>/           Curves for each optimizer ablation variant
      optimizer_ablation_summary.png
      optimizer_ablation_summary.csv
    mlp/
      <ablation_name>/           Curves for each MLP ablation variant
      mlp_ablation_summary.png
      mlp_ablation_summary.csv
"""

import os
import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATA LOADING
# ============================================================
def load_csv_dataset(data_dir: str = "dataset") -> Tuple[torch.Tensor, ...]:
    """Load Fashion-MNIST from local CSV files into PyTorch tensors."""
    train_data = np.loadtxt(os.path.join(data_dir, "fashion-mnist_train.csv"),
                            delimiter=",", skiprows=1)
    test_data  = np.loadtxt(os.path.join(data_dir, "fashion-mnist_test.csv"),
                            delimiter=",", skiprows=1)

    X_train = torch.tensor(train_data[:, 1:].astype(np.float32) / 255.0)
    y_train = torch.tensor(train_data[:, 0].astype(np.int64), dtype=torch.long)
    X_test  = torch.tensor(test_data[:, 1:].astype(np.float32) / 255.0)
    y_test  = torch.tensor(test_data[:, 0].astype(np.int64),  dtype=torch.long)

    return X_train, y_train, X_test, y_test


def make_loader(X: torch.Tensor, y: torch.Tensor,
                batch_size: int, shuffle: bool = True,
                seed: int = 0) -> DataLoader:
    dataset = TensorDataset(X.to(DEVICE), y.to(DEVICE))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, generator=g)


# ============================================================
# HELPERS
# ============================================================
def tensor_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def confidence_interval(values: List[float]) -> Tuple[float, float, float]:
    arr  = np.array(values)
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1))
    ci   = 1.96 * std / np.sqrt(len(arr))
    return mean, std, ci


def _lr_tag(lr: float) -> str:
    return f"lr_{lr:.0e}".replace("+", "").replace("-0", "-")


# ============================================================
# MODEL CONFIGURATION
# ============================================================
@dataclass
class MLPConfig:
    """All knobs that define the MLP architecture."""
    hidden_dims:  List[int]  = field(default_factory=lambda: [256])
    activation:   str        = "relu"      # relu | tanh | gelu | leaky_relu
    init_scheme:  str        = "he"        # he | xavier | zeros
    dropout:      float      = 0.0
    batch_norm:   bool       = False
    input_dim:    int        = 784
    output_dim:   int        = 10

    @property
    def label(self) -> str:
        parts = [
            f"h={'_'.join(str(d) for d in self.hidden_dims)}",
            self.activation,
            self.init_scheme,
            f"drop={self.dropout}",
            f"bn={int(self.batch_norm)}",
        ]
        return "__".join(parts)


@dataclass
class OptimizerConfig:
    """All knobs that define the optimizer."""
    name:         str   = "adam"    # adam | sgd
    lr:           float = 1e-3
    momentum:     float = 0.9       # SGD only
    weight_decay: float = 0.0
    # Adam-specific
    beta1:        float = 0.9
    beta2:        float = 0.999
    eps:          float = 1e-8
    amsgrad:      bool  = False

    @property
    def label(self) -> str:
        if self.name == "sgd":
            return (f"sgd__lr={self.lr:.0e}__mom={self.momentum}"
                    f"__wd={self.weight_decay}")
        return (f"adam__lr={self.lr:.0e}__b1={self.beta1}__b2={self.beta2}"
                f"__eps={self.eps:.0e}__wd={self.weight_decay}"
                f"__amsgrad={int(self.amsgrad)}")


# ============================================================
# MODEL FACTORY
# ============================================================
class FlexMLP(nn.Module):
    """
    Configurable MLP: variable depth, activation, init, dropout, batch norm.
    """

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg

        dims   = [cfg.input_dim] + cfg.hidden_dims
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(out_d))
            layers.append(self._act())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))

        layers.append(nn.Linear(dims[-1], cfg.output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _act(self) -> nn.Module:
        return {
            "relu":       nn.ReLU(),
            "tanh":       nn.Tanh(),
            "gelu":       nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }[self.cfg.activation]

    def _init_weights(self):
        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue
            s = self.cfg.init_scheme
            if s == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif s == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif s == "zeros":
                nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model(seed: int, mlp_cfg: Optional[MLPConfig] = None) -> FlexMLP:
    torch.manual_seed(seed)
    cfg = mlp_cfg if mlp_cfg is not None else MLPConfig()
    return FlexMLP(cfg).to(DEVICE)


def make_optimizer(model: nn.Module, opt_cfg: OptimizerConfig) -> torch.optim.Optimizer:
    if opt_cfg.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg.lr,
        betas=(opt_cfg.beta1, opt_cfg.beta2),
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
        amsgrad=opt_cfg.amsgrad,
    )


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model: nn.Module,
             X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(X.to(DEVICE))
        loss   = F.cross_entropy(logits, y.to(DEVICE)).item()
        preds  = logits.argmax(dim=1)
        acc    = tensor_accuracy(preds, y.to(DEVICE))
    return loss, acc, 1.0 - acc


# ============================================================
# CORE TRAINING LOOP
# ============================================================
def train_model(opt_cfg: OptimizerConfig,
                X_train, y_train, X_val, y_val, X_test, y_test,
                seed: int,
                mlp_cfg: Optional[MLPConfig] = None,
                epochs: int = 500,
                batch_size: int = 1024) -> Tuple:
    """
    Train one (MLPConfig, OptimizerConfig) combination.
    Returns (model_at_best_val, best_val_acc, history).
    """
    model  = make_model(seed, mlp_cfg)
    opt    = make_optimizer(model, opt_cfg)
    loader = make_loader(X_train, y_train, batch_size, shuffle=True, seed=seed)

    history: Dict[str, List[float]] = {
        k: [] for k in ("train_loss", "train_acc", "train_error",
                         "val_acc", "test_loss", "test_acc", "test_error")
    }
    best_val  = -1.0
    best_state: Optional[dict] = None

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()

        tr_loss, tr_acc, tr_err = evaluate(model, X_train, y_train)
        _,       val_acc, _     = evaluate(model, X_val,   y_val)
        te_loss, te_acc, te_err = evaluate(model, X_test,  y_test)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_error"].append(tr_err)
        history["val_acc"].append(val_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["test_error"].append(te_err)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_val, history


# ============================================================
# MAIN EXPERIMENT — GRID SEARCH
# ============================================================
def run_grid_search(optimizer_name: str, lr_grid: List[float], seed: int,
                    X_train, y_train, X_val, y_val, X_test, y_test):
    """Run LR grid search for one optimizer using the default MLP config."""
    best_model, best_lr, best_val_acc, best_history = None, None, -1.0, None
    all_histories: Dict[float, Dict] = {}

    for lr in lr_grid:
        opt_cfg = OptimizerConfig(name=optimizer_name, lr=lr)
        print(f"  [{optimizer_name.upper()}] Seed={seed}  LR={lr:.0e}")
        model, val_acc, history = train_model(
            opt_cfg, X_train, y_train, X_val, y_val, X_test, y_test, seed)
        all_histories[lr] = history

        fl, fa, fe = history["train_loss"][-1], history["train_acc"][-1], history["train_error"][-1]
        tl, ta, te = history["test_loss"][-1],  history["test_acc"][-1],  history["test_error"][-1]
        va         = history["val_acc"][-1]
        print(f"    TrainLoss={fl:.4f} TrainAcc={fa:.4f} TrainErr={fe:.4f} "
              f"TestLoss={tl:.4f} TestAcc={ta:.4f} TestErr={te:.4f} ValAcc={va:.4f}")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_model    = copy.deepcopy(model)
            best_lr       = lr
            best_history  = history

    return best_model, best_lr, best_val_acc, best_history, all_histories


# ============================================================
# PLOTTING
# ============================================================
_PLOT_SPECS = [
    ("train_loss",  "Loss",     "Training Loss"),
    ("train_acc",   "Accuracy", "Training Accuracy"),
    ("train_error", "Error",    "Training Error"),
    ("val_acc",     "Accuracy", "Validation Accuracy"),
    ("test_loss",   "Loss",     "Test Loss"),
    ("test_acc",    "Accuracy", "Test Accuracy"),
    ("test_error",  "Error",    "Test Error"),
]


def _save_dual_plot(out_path, epochs,
                    vals_a, label_a,
                    vals_b, label_b,
                    ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, vals_a, label=label_a)
    ax.plot(epochs, vals_b, label=label_b)
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_combined(adam_history, sgd_history, seed, adam_lr, sgd_lr,
                  base_dir="torch_plots"):
    out_dir = os.path.join(base_dir, f"seed_{seed}_combined")
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(adam_history["train_loss"]) + 1)
    for key, ylabel, title in _PLOT_SPECS:
        _save_dual_plot(
            os.path.join(out_dir, f"{key}.png"), epochs,
            adam_history[key], f"Adam (lr={adam_lr:.0e})",
            sgd_history[key],  f"SGD  (lr={sgd_lr:.0e})",
            ylabel, f"{title} — Seed {seed}")


def plot_per_lr(all_adam_histories, all_sgd_histories, seed,
                base_dir="torch_plots"):
    for lr in set(all_adam_histories) | set(all_sgd_histories):
        has_adam = lr in all_adam_histories
        has_sgd  = lr in all_sgd_histories
        tag = "both" if (has_adam and has_sgd) else ("adam" if has_adam else "sgd")
        folder = os.path.join(base_dir, tag, _lr_tag(lr), f"seed_{seed}")
        os.makedirs(folder, exist_ok=True)
        n = max(
            len(all_adam_histories[lr]["train_loss"]) if has_adam else 0,
            len(all_sgd_histories[lr]["train_loss"])  if has_sgd  else 0,
        )
        epochs = range(1, n + 1)
        for key, ylabel, title_prefix in _PLOT_SPECS:
            fig, ax = plt.subplots(figsize=(8, 6))
            if has_adam:
                ax.plot(epochs, all_adam_histories[lr][key], label="Adam")
            if has_sgd:
                ax.plot(epochs, all_sgd_histories[lr][key],  label="SGD")
            ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.set_title(f"{title_prefix} — LR={lr:.0e}, Seed {seed}")
            ax.legend(); fig.tight_layout()
            fig.savefig(os.path.join(folder, f"{key}.png")); plt.close(fig)


def plot_confidence_bands(adam_histories, sgd_histories, base_dir="torch_plots"):
    out_dir = os.path.join(base_dir, "confidence_bands")
    os.makedirs(out_dir, exist_ok=True)
    for key, ylabel, title in _PLOT_SPECS:
        am = np.array([h[key] for h in adam_histories])
        sm = np.array([h[key] for h in sgd_histories])
        ep = np.arange(1, am.shape[1] + 1)
        a_mean, s_mean = am.mean(0), sm.mean(0)
        a_ci = 1.96 * am.std(0, ddof=1) / np.sqrt(len(adam_histories))
        s_ci = 1.96 * sm.std(0, ddof=1) / np.sqrt(len(sgd_histories))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ep, a_mean, label="Adam"); ax.plot(ep, s_mean, label="SGD")
        ax.fill_between(ep, a_mean - a_ci, a_mean + a_ci, alpha=0.25)
        ax.fill_between(ep, s_mean - s_ci, s_mean + s_ci, alpha=0.25)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (Mean ± 95% CI Across Seeds)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{key}_confidence_band.png"))
        plt.close(fig)


# ============================================================
# ABLATION ENGINE
# ============================================================
def run_ablation(
    variants: List[Dict[str, Any]],
    X_train, y_train, X_val, y_val, X_test, y_test,
    seeds: List[int],
    out_dir: str,
    epochs: int = 200,
    batch_size: int = 1024,
) -> List[Dict]:
    """
    Generic ablation runner.

    Each variant is a dict with keys:
      label      : str — short human-readable name shown in plots/CSV
      opt_cfg    : OptimizerConfig
      mlp_cfg    : MLPConfig  (optional; defaults to baseline MLPConfig)

    Returns a list of result dicts with mean / CI across seeds.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary: List[Dict] = []

    for v in variants:
        label    = v["label"]
        opt_cfg: OptimizerConfig          = v["opt_cfg"]
        mlp_cfg: Optional[MLPConfig]      = v.get("mlp_cfg", None)

        print(f"\n  [ABLATION] {label}")
        seed_test_accs: List[float] = []
        seed_histories: List[Dict]  = []

        for seed in seeds:
            model, _, history = train_model(
                opt_cfg, X_train, y_train, X_val, y_val, X_test, y_test,
                seed, mlp_cfg, epochs=epochs, batch_size=batch_size,
            )
            model.eval()
            with torch.no_grad():
                test_acc = tensor_accuracy(
                    model(X_test.to(DEVICE)).argmax(1), y_test.to(DEVICE))
            seed_test_accs.append(test_acc)
            seed_histories.append(history)
            print(f"    Seed {seed}: test_acc={test_acc:.4f}")

        mean, std, ci = confidence_interval(seed_test_accs)
        summary.append({"label": label, "mean": mean, "std": std, "ci": ci})

        # Per-variant learning curves (mean ± CI across seeds)
        var_dir = os.path.join(out_dir, label.replace(" ", "_").replace("/", "-"))
        os.makedirs(var_dir, exist_ok=True)
        _plot_ablation_curves(seed_histories, label, var_dir)

    return summary


def _plot_ablation_curves(seed_histories: List[Dict], label: str, out_dir: str):
    """Mean ± 95% CI learning curves for a single ablation variant."""
    for key, ylabel, title_prefix in _PLOT_SPECS:
        mat    = np.array([h[key] for h in seed_histories])
        ep     = np.arange(1, mat.shape[1] + 1)
        mean   = mat.mean(0)
        ci     = 1.96 * mat.std(0, ddof=1) / np.sqrt(len(seed_histories))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ep, mean, label=label)
        ax.fill_between(ep, mean - ci, mean + ci, alpha=0.25)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} — {label} (Mean ± 95% CI)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{key}.png")); plt.close(fig)


def save_ablation_summary(summary: List[Dict], out_dir: str, name: str):
    """Bar chart + CSV for a completed ablation sweep."""
    labels = [r["label"] for r in summary]
    means  = [r["mean"]  for r in summary]
    cis    = [r["ci"]    for r in summary]

    # Bar chart
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 6))
    bars = ax.bar(x, means, yerr=cis, capsize=5, color="steelblue", alpha=0.8)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"{name} — Test Accuracy (Mean ± 95% CI)")
    ax.set_ylim(max(0, min(means) - 0.05), min(1.0, max(means) + 0.08))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_summary.png"))
    plt.close(fig)

    # CSV
    csv_path = os.path.join(out_dir, f"{name}_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "mean", "std", "ci"])
        w.writeheader()
        w.writerows(summary)
    print(f"\n  Summary saved → {csv_path}")


# ============================================================
# ABLATION DEFINITIONS
# ============================================================
def build_optimizer_ablations(best_adam_lr: float,
                               best_sgd_lr: float) -> List[Dict]:
    """
    Isolate one optimizer hyperparameter at a time while holding
    all others at their baseline values.
    """
    variants: List[Dict] = []

    # ── Baseline configs ──────────────────────────────────────
    variants.append({"label": "Adam_baseline",
                     "opt_cfg": OptimizerConfig(name="adam", lr=best_adam_lr)})
    variants.append({"label": "SGD_baseline",
                     "opt_cfg": OptimizerConfig(name="sgd",  lr=best_sgd_lr)})

    # ── SGD: momentum sweep ───────────────────────────────────
    for mom in [0.0, 0.5, 0.99]:
        variants.append({
            "label":   f"SGD_mom={mom}",
            "opt_cfg": OptimizerConfig(name="sgd", lr=best_sgd_lr, momentum=mom),
        })

    # ── SGD: weight decay ─────────────────────────────────────
    for wd in [1e-5, 1e-4, 1e-3]:
        variants.append({
            "label":   f"SGD_wd={wd:.0e}",
            "opt_cfg": OptimizerConfig(name="sgd", lr=best_sgd_lr, weight_decay=wd),
        })

    # ── Adam: β₁ sweep ───────────────────────────────────────
    for b1 in [0.5, 0.8, 0.95, 0.99]:
        variants.append({
            "label":   f"Adam_b1={b1}",
            "opt_cfg": OptimizerConfig(name="adam", lr=best_adam_lr, beta1=b1),
        })

    # ── Adam: β₂ sweep ───────────────────────────────────────
    for b2 in [0.9, 0.99, 0.9999]:
        variants.append({
            "label":   f"Adam_b2={b2}",
            "opt_cfg": OptimizerConfig(name="adam", lr=best_adam_lr, beta2=b2),
        })

    # ── Adam: ε sweep ────────────────────────────────────────
    for eps in [1e-10, 1e-6, 1e-4]:
        variants.append({
            "label":   f"Adam_eps={eps:.0e}",
            "opt_cfg": OptimizerConfig(name="adam", lr=best_adam_lr, eps=eps),
        })

    # ── Adam: weight decay ───────────────────────────────────
    for wd in [1e-5, 1e-4, 1e-3]:
        variants.append({
            "label":   f"Adam_wd={wd:.0e}",
            "opt_cfg": OptimizerConfig(name="adam", lr=best_adam_lr, weight_decay=wd),
        })

    # ── Adam: AMSGrad ─────────────────────────────────────────
    variants.append({
        "label":   "Adam_AMSGrad",
        "opt_cfg": OptimizerConfig(name="adam", lr=best_adam_lr, amsgrad=True),
    })

    return variants


def build_mlp_ablations(best_adam_lr: float) -> List[Dict]:
    """
    Isolate one MLP architectural choice at a time.
    All runs use Adam at the best LR found in the main experiment.
    """
    base_opt = OptimizerConfig(name="adam", lr=best_adam_lr)
    variants: List[Dict] = []

    # ── Baseline ──────────────────────────────────────────────
    variants.append({
        "label":   "MLP_baseline",
        "opt_cfg": base_opt,
        "mlp_cfg": MLPConfig(),
    })

    # ── Hidden dimension ──────────────────────────────────────
    for hd in [64, 128, 512, 1024]:
        variants.append({
            "label":   f"hidden={hd}",
            "opt_cfg": base_opt,
            "mlp_cfg": MLPConfig(hidden_dims=[hd]),
        })

    # ── Depth (number of hidden layers) ──────────────────────
    for depth, dims in [(1, [256]), (2, [256, 256]), (3, [256, 256, 256]),
                        (4, [256, 256, 256, 256])]:
        if depth == 1:
            continue  # already the baseline
        variants.append({
            "label":   f"depth={depth}",
            "opt_cfg": base_opt,
            "mlp_cfg": MLPConfig(hidden_dims=dims),
        })

    # ── Activation function ───────────────────────────────────
    for act in ["tanh", "gelu", "leaky_relu"]:
        variants.append({
            "label":   f"act={act}",
            "opt_cfg": base_opt,
            "mlp_cfg": MLPConfig(activation=act),
        })

    # ── Weight initialisation ─────────────────────────────────
    for init in ["xavier", "zeros"]:
        variants.append({
            "label":   f"init={init}",
            "opt_cfg": base_opt,
            "mlp_cfg": MLPConfig(init_scheme=init),
        })

    # ── Dropout ───────────────────────────────────────────────
    for dr in [0.2, 0.5]:
        variants.append({
            "label":   f"dropout={dr}",
            "opt_cfg": base_opt,
            "mlp_cfg": MLPConfig(dropout=dr),
        })

    # ── Batch normalisation ───────────────────────────────────
    variants.append({
        "label":   "batch_norm=True",
        "opt_cfg": base_opt,
        "mlp_cfg": MLPConfig(batch_norm=True),
    })

    # ── Combined: BN + Dropout ────────────────────────────────
    variants.append({
        "label":   "bn+drop=0.2",
        "opt_cfg": base_opt,
        "mlp_cfg": MLPConfig(batch_norm=True, dropout=0.2),
    })

    # ── Combined: wide + deep ─────────────────────────────────
    variants.append({
        "label":   "wide_deep",
        "opt_cfg": base_opt,
        "mlp_cfg": MLPConfig(hidden_dims=[512, 512, 256]),
    })

    return variants


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Using device: {DEVICE}")

    X_train_full, y_train_full, X_test, y_test = load_csv_dataset()
    X_train = X_train_full[:50000];  y_train = y_train_full[:50000]
    X_val   = X_train_full[50000:];  y_val   = y_train_full[50000:]

    seeds     = [0, 1, 2, 3, 4]
    adam_grid = [1e-4, 3e-4, 1e-3, 3e-3]
    sgd_grid  = [1e-3, 1e-2, 5e-2, 1e-1]

    adam_results:        List[float] = []
    sgd_results:         List[float] = []
    adam_seed_histories: List[Dict]  = []
    sgd_seed_histories:  List[Dict]  = []

    # Track best LRs across seeds for ablation anchoring
    adam_lr_votes: List[float] = []
    sgd_lr_votes:  List[float] = []

    # ── Main experiment ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("MAIN EXPERIMENT")
    print("=" * 60)

    for seed in seeds:
        print(f"\nSeed {seed}")

        adam_model, adam_lr, _, adam_history, all_adam = run_grid_search(
            "adam", adam_grid, seed, X_train, y_train, X_val, y_val, X_test, y_test)
        sgd_model,  sgd_lr,  _, sgd_history,  all_sgd  = run_grid_search(
            "sgd",  sgd_grid,  seed, X_train, y_train, X_val, y_val, X_test, y_test)

        adam_lr_votes.append(adam_lr)
        sgd_lr_votes.append(sgd_lr)

        adam_model.eval(); sgd_model.eval()
        with torch.no_grad():
            adam_test_acc = tensor_accuracy(
                adam_model(X_test.to(DEVICE)).argmax(1), y_test.to(DEVICE))
            sgd_test_acc  = tensor_accuracy(
                sgd_model(X_test.to(DEVICE)).argmax(1),  y_test.to(DEVICE))

        adam_results.append(adam_test_acc)
        sgd_results.append(sgd_test_acc)
        adam_seed_histories.append(adam_history)
        sgd_seed_histories.append(sgd_history)

        plot_combined(adam_history, sgd_history, seed, adam_lr, sgd_lr)
        plot_per_lr(all_adam, all_sgd, seed)

        print(f"  Adam best LR={adam_lr:.0e}  Test Acc={adam_test_acc:.4f}")
        print(f"  SGD  best LR={sgd_lr:.0e}   Test Acc={sgd_test_acc:.4f}")

    plot_confidence_bands(adam_seed_histories, sgd_seed_histories)

    adam_mean, _, adam_ci = confidence_interval(adam_results)
    sgd_mean,  _, sgd_ci  = confidence_interval(sgd_results)

    print("\nFinal Results")
    print("=" * 60)
    print(f"Adam Mean Test Accuracy: {adam_mean:.4f} ± {adam_ci:.4f}")
    print(f"SGD  Mean Test Accuracy: {sgd_mean:.4f}  ± {sgd_ci:.4f}")
    print(f"Generalization Gap (SGD - Adam): {sgd_mean - adam_mean:.4f}")

    # Anchor ablations to the most-selected LR across seeds
    from collections import Counter
    best_adam_lr = Counter(adam_lr_votes).most_common(1)[0][0]
    best_sgd_lr  = Counter(sgd_lr_votes).most_common(1)[0][0]
    print(f"\nAblation anchor LRs — Adam: {best_adam_lr:.0e}  SGD: {best_sgd_lr:.0e}")

    # ── Optimizer ablation ────────────────────────────────────
    print("\n" + "=" * 60)
    print("OPTIMIZER ABLATION")
    print("=" * 60)

    opt_variants = build_optimizer_ablations(best_adam_lr, best_sgd_lr)
    opt_out      = os.path.join("torch_plots", "ablation", "optimizer")
    opt_summary  = run_ablation(
        opt_variants, X_train, y_train, X_val, y_val, X_test, y_test,
        seeds, opt_out, epochs=200)
    save_ablation_summary(opt_summary, opt_out, "optimizer_ablation")

    # ── MLP ablation ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MLP ABLATION")
    print("=" * 60)

    mlp_variants = build_mlp_ablations(best_adam_lr)
    mlp_out      = os.path.join("torch_plots", "ablation", "mlp")
    mlp_summary  = run_ablation(
        mlp_variants, X_train, y_train, X_val, y_val, X_test, y_test,
        seeds, mlp_out, epochs=200)
    save_ablation_summary(mlp_summary, mlp_out, "mlp_ablation")

    # ── Ablation analysis printout ────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION ANALYSIS")
    print("=" * 60)

    for group_name, summary in [("Optimizer", opt_summary), ("MLP", mlp_summary)]:
        print(f"\n{group_name} ablation results (sorted by mean test accuracy):")
        ranked = sorted(summary, key=lambda r: r["mean"], reverse=True)
        baseline = next((r for r in ranked if "baseline" in r["label"]), ranked[0])
        for rank, r in enumerate(ranked, 1):
            delta = r["mean"] - baseline["mean"]
            marker = " ◀ baseline" if r["label"] == baseline["label"] else \
                     f" ({delta:+.4f} vs baseline)"
            print(f"  {rank:2d}. {r['label']:<30s}  "
                  f"{r['mean']:.4f} ± {r['ci']:.4f}{marker}")


if __name__ == "__main__":
    main()