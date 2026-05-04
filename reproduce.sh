#!/usr/bin/env bash

# ============================================================
# Reproduction Script
# ============================================================
# Regenerates all figures and outputs for the full project
# from a clean repository clone.
#
# Outputs:
# - NumPy experiment figures
# - PyTorch experiment figures
# - Confidence interval plots
# - Ablation plots
# - Synthetic Section 3 plots
# ============================================================

set -e  # Exit immediately on failure

echo "============================================================"
echo "Optimizer Generalization Reproduction"
echo "Full Reproduction Pipeline"
echo "============================================================"

# ------------------------------------------------------------
# 1. Create virtual environment (optional but recommended)
# ------------------------------------------------------------
if [ ! -d "venv" ]; then
    echo "[1/6] Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# ------------------------------------------------------------
# 2. Install dependencies
# ------------------------------------------------------------
echo "[2/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ------------------------------------------------------------
# 3. Prepare output directories
# ------------------------------------------------------------
echo "[3/6] Preparing output directories..."
rm -rf numpy_plots torch_plots synthetic_section3_outputs
mkdir -p numpy_plots
mkdir -p torch_plots
mkdir -p synthetic_section3_outputs

# ------------------------------------------------------------
# 4. Run NumPy experiment
# ------------------------------------------------------------
echo "[4/6] Running NumPy experiment..."
python numpy_experiment.py

echo "NumPy experiment complete."

# ------------------------------------------------------------
# 5. Run PyTorch experiment + ablations
# ------------------------------------------------------------
echo "[5/6] Running PyTorch experiment + ablations..."
python pytorch_experiment.py

echo "PyTorch experiment complete."

# ------------------------------------------------------------
# 6. Run synthetic Section 3 reproduction
# ------------------------------------------------------------
echo "[6/6] Running synthetic Section 3 reproduction..."
python synthetic_adaptive_vs_sgd.py

# Move synthetic output into dedicated folder if generated at root
if [ -f "synthetic_section3_convergence.png" ]; then
    mv synthetic_section3_convergence.png synthetic_section3_outputs/
fi

# ------------------------------------------------------------
# Done
# ------------------------------------------------------------
echo "============================================================"
echo "Reproduction complete."
echo "Generated outputs:"
echo "  - ./numpy_plots"
echo "  - ./torch_plots"
echo "  - ./synthetic_section3_outputs"
echo "============================================================"
