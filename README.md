# SechKAN_1D

Official implementation for the paper:

> **Evaluating the Efficacy of SechKAN on 1D Data**, submitted to [SOICT 2026](https://soict.org/).

This repository provides the code and experimental configurations for evaluating **SechKAN** on one-dimensional classification datasets. The experiments compare SechKAN with conventional neural networks, 1D convolutional architectures, and other KAN-based models.

## Overview

The repository evaluates the following models:

* **MLP** — Multilayer Perceptron
* **CNN-1D** — One-dimensional Convolutional Neural Network
* **ResNet-1D** — One-dimensional Residual Network
* **DS-CNN-1D** — Depthwise-Separable CNN for 1D data
* **EfficientKAN** — Efficient Kolmogorov-Arnold Network
* **SechKAN** — Kolmogorov-Arnold Network using hyperbolic secant basis functions

---

## Repository Structure

```text
SechKAN_1D/
│
├── data/
│   └── ...                         # Dataset files
│
├── models/
│   ├── ...                         # Model implementations
│
├── run_har.py                      # UCI HAR experiments
├── run_crop.py                     # Crop experiments
├── run_ele_dev.py                  # ElectricDevices experiments
│
├── run_har_ex.sh                   # HAR experiment commands
├── run_crop_ex.sh                  # Crop experiment commands
├── run_ele_dev.sh                  # ElectricDevices experiment commands
│
├── README.md
└── ...
```

---

# Datasets

The experiments use three one-dimensional classification datasets:

* **UCI HAR** — Human activity recognition using 561-dimensional sensor feature vectors (`run_har.py`).
* **Crop** — One-dimensional time-series classification dataset (`run_crop.py`).
* **ElectricDevices** — One-dimensional time-series classification dataset from the UCR Time Series Classification Archive (`run_ele_dev.py`).

Place the datasets under the `data/` directory according to the structure expected by the corresponding scripts.


# Running the Experiments

The three main experiment scripts are:

```text
run_har.py
run_crop.py
run_ele_dev.py
```

The corresponding shell scripts contain the experimental configurations used in the study:

```text
run_har_ex.sh
run_crop_ex.sh
run_ele_dev.sh
```

For the complete experimental settings, please refer to these shell scripts.

---

# Common Parameters

The following command-line parameters are shared across the three experiment scripts.

* `--model`: Select the model to train. Available options are `mlp`, `cnn_1d`, `resnet_1d`, `ds_cnn_1d`, `sech_kan`, and `efficient_kan`.
* `--data_root`: Root directory containing the datasets. Default: `./data`.
* `--note`: Optional note or identifier for the experiment run. Default: empty string.
* `--val_subject_fraction`: Fraction of subjects used for validation when subject-level validation is applicable. Default: `0.2`.
* `--batch_size`: Training batch size. Default: `64`.
* `--epochs`: Number of training epochs. Default: `20`.
* `--hidden_layers`: Hidden-layer configuration of the model. Default: `"256"`.
* `--lr`: Learning rate. Default: `1e-3`.
* `--weight_decay`: Weight decay used by the optimizer. Default: `1e-4`.
* `--scheduler`: Learning-rate scheduler. Available options are `StepLR`, `CosineAnnealingLR`, `OneCycleLR`, `ExponentialLR`, and `CyclicLR`. Default: `OneCycleLR`.
* `--ratio`: Dimensionality-reduction ratio used by the pairwise reduction mechanism. Default: `2`.
* `--pairwise_fn`: Pairwise function used for feature reduction. Default: `atan`.
* `--pairing`: Feature-pairing strategy. Available options are `original`, `shuffle`, and `reverse`. Default: `original`.
* `--pair_type`: Pairwise reduction operation. Default: `raw_weighted_sum_product`.
* `--norm1_type`: Normalization type used in the first normalization stage. Default: empty.
* `--norm2_type`: Normalization type used in the second normalization stage. Default: `layer`.
* `--norm_mode`: Specifies where normalization is applied. Available options are `none`, `first`, `except_first`, and `all`. Default: `all`.
* `--norm_type`: General normalization type used by models that require this setting. Default: `layer`.
* `--activation`: Activation function used by the model. Default: `silu`.
* `--num_workers`: Number of workers used for data loading. Default: `4`.
* `--seed`: Random seed used for reproducibility. Default: `42`.
* `--device`: Computing device. Default: `cuda`.
* `--num_grids`: Number of grid points used by SechKAN. Default: `4`.

---

# Example Commands

Check `run_har_ex.sh`, `run_crop_ex.sh`, and `run_ele_dev.sh` for the complete experimental configurations.

### ElectricDevices

```bash
python run_ele_dev.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0"
```

### UCI HAR

```bash
python run_har.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0"
```

### Crop

```bash
python run_crop.py --model "sech_kan" --num_grids 16 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "batch" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0"
```

---

# Reproducibility

To reproduce the experiments reported in the paper, use the provided shell scripts:

```text
run_har_ex.sh
run_crop_ex.sh
run_ele_dev.sh
```

These scripts contain the model configurations, hyperparameters, random seeds, and other experimental settings used in the study.
