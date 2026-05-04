# Optimizer Research Validation

This repository reproduces and extends experiments from the paper:

***The Marginal Value of Adaptive Gradient Methods in Machine Learning***
by entity["people","Ashia C. Wilson","machine learning researcher"] et al. (2017)

The project investigates optimizer generalization behavior by comparing adaptive methods (Adam/AdaGrad) against SGD with momentum on the entity["dataset","Fashion-MNIST","image classification dataset"] benchmark, and reproduces the paper’s synthetic overparameterized construction demonstrating optimizer-dependent implicit bias.

---

## Project Overview

This repository contains three major implementations:

### 1. NumPy Implementation

A full manual implementation of a 2-layer MLP using only NumPy.

Includes:

* Forward propagation
* Backpropagation
* Manual parameter updates
* Adam optimizer
* SGD + Momentum optimizer
* Training/validation/testing pipeline
* Confidence interval analysis across 5 random seeds

Purpose:

* Demonstrate low-level optimizer mechanics
* Compare optimizer behavior without framework abstractions

---

### 2. PyTorch Implementation + Ablation Study

A framework-based reproduction using PyTorch.

Includes:

* Flexible MLP architecture
* Optimizer grid search
* Adam vs SGD comparison
* 95% confidence interval reporting
* Optimizer ablation studies
* MLP architecture ablation studies
* Automatic plotting/export pipeline

Ablation dimensions:

Optimizer:

* Momentum
* Weight decay
* Adam betas
* Adam epsilon
* AMSGrad

MLP:

* Width
* Depth
* Activation functions
* Dropout
* Batch normalization
* Initialization strategy

Purpose:

* Reproduce the paper more efficiently
* Analyze optimizer and architecture sensitivity

---

### 3. Section 3 Synthetic Construction

A synthetic reproduction of the paper’s overparameterized linear regression example.

Includes:

* Minimum-norm solution derivation
* Gradient descent comparison
* AdaGrad comparison
* Convergence visualization

Purpose:

* Demonstrate optimizer-dependent implicit bias
* Show why adaptive methods may converge to worse solutions despite identical training loss

---

## Repository Structure

```text
project/
│── numpy_experiment.py
│── pytorch_experiment.py
│── synthetic_adaptive_vs_sgd.py
│── dataset/
│   ├── fashion-mnist_train.csv
│   └── fashion-mnist_test.csv
│── numpy_plots/
│── torch_plots/
│── report/
│── README.md
```

---

## Dataset

This project uses entity["dataset","Fashion-MNIST","image classification dataset"].

Dataset format:

```text
label,pixel1,pixel2,...,pixel784
```

Input dimensions:

* 784 features (28×28 grayscale image)

Classes:

* 10 clothing categories

Normalization:

* Pixel values scaled to [0,1]

---

## Model Architecture

Baseline MLP:

```text
Input (784)
   ↓
Linear (784 → 256)
   ↓
ReLU
   ↓
Linear (256 → 10)
   ↓
Softmax / Cross Entropy
```

Loss Function:

```text
CrossEntropyLoss
```

---

## Optimizers Compared

### Adam

Adaptive gradient optimization with moment estimation.

Parameters:

* β₁ = 0.9
* β₂ = 0.999
* ε = 1e-8

---

### SGD + Momentum

Standard stochastic gradient descent with momentum.

Parameters:

* Momentum = 0.9

---

## Experimental Procedure

1. Load Fashion-MNIST CSV dataset
2. Split into train/validation/test sets
3. Perform learning-rate grid search
4. Train models for multiple random seeds
5. Select best validation checkpoint
6. Evaluate on test set
7. Compute mean and 95% confidence intervals
8. Run ablation studies
9. Generate plots

---

## Key Findings

### Optimizer Findings

* Adam and SGD achieved similar final performance.
* Adam converged faster.
* SGD often generalized comparably or slightly better under tuned settings.
* Weight decay improved both optimizers.
* Extreme momentum harmed SGD performance.

### Architecture Findings

* Larger/deeper models improved performance more than optimizer tuning.
* Dropout improved generalization.
* Zero initialization caused complete training failure.

### Synthetic Findings

* Gradient descent converged to the minimum-norm solution.
* Adaptive methods converged to a different implicit solution.
* This supports the paper’s Section 3 theoretical claims.

---

## Installation

Install dependencies:

```bash
pip install numpy matplotlib torch
```

---

## Running the Experiments

### NumPy Version

```bash
python numpy_experiment.py
```

### PyTorch Version

```bash
python pytorch_experiment.py
```

### Synthetic Reproduction

```bash
python synthetic_adaptive_vs_sgd.py
```

---

## Output

Generated outputs include:

* Training loss plots
* Test accuracy plots
* Confidence interval plots
* Ablation summary plots
* Synthetic convergence plots
* CSV summaries

---

## Reference

Wilson, A. C., Roelofs, R., Stern, M., Srebro, N., & Recht, B. (2017).
*The Marginal Value of Adaptive Gradient Methods in Machine Learning.*

---

## License

This project is intended for academic and educational use.
