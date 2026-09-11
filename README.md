# WeirNet: A Large-Scale 3D CFD Benchmark for Geometric Surrogate Modeling of Piano Key Weirs

## Abstract

Reliable prediction of hydraulic performance is challenging for Piano Key Weir (PKW) design because discharge capacity depends on three-dimensional geometry and operating conditions. Surrogate models can accelerate hydraulic-structure design, but progress is limited by scarce large, well-documented datasets that jointly capture geometric variation, operating conditions, and functional performance. This study presents WeirNet, a large 3D CFD benchmark dataset for geometric surrogate modeling of PKWs. WeirNet contains 3,794 parametric, feasibility-constrained rectangular and trapezoidal PKW geometries, each scheduled at 19 discharge conditions using a consistent free-surface OpenFOAM workflow, resulting in 71,387 completed simulations that form the benchmark and with complete discharge coefficient labels. The dataset is released as multiple modalities compact parametric descriptors, watertight surface meshes and high-resolution point clouds together with standardized tasks and in-distribution and out-of-distribution splits. Representative surrogate families are benchmarked for discharge coefficient prediction. Tree-based regressors on parametric descriptors achieve the best overall accuracy, while point- and mesh-based models remain competitive and offer parameterization-agnostic inference. All surrogates evaluate in milliseconds per sample, providing orders-of-magnitude speedups over CFD runtimes. Out-of-distribution results identify geometry shift as the dominant failure mode compared to unseen discharge values, and data-efficiency experiments show diminishing returns beyond roughly 60% of the training data. By publicly releasing the dataset together with simulation setups and evaluation pipelines, WeirNet establishes a reproducible framework for data-driven hydraulic modeling and enables faster exploration of PKW designs during the early stages of hydraulic planning.

## Dataset

Download the dataset from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22707711.svg)](https://doi.org/10.5281/zenodo.22707711)

Place the extracted files in the following structure:

```text
Data/PKW_Efficiency_Dataset/
├── combined_data.npz
├── pc/
└── stl/
```

For another location, set `--data_path`, `--pc_dir`, and `--stl_dir` when running the neural models.

## Setup

Create and activate the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate weirnet_env
```

The [environment.yml](environment.yml) file includes Python 3.11, PyTorch with CUDA 12.1, and the dependencies for training and preprocessing.

## Training

Run commands from the repository root:

```bash
# PointNet (use --model DGCNN for DGCNN)
python train_ddp.py --model PN --epochs 100 --num_workers 4

# Graph neural network
python trainGNN.py --model GNN --epochs 100 --num_workers 4

# Tabular regression baselines
python train_regressors.py
```

Neural model checkpoints are saved under `saved_model/`. Add `--resume` to continue training, or use `--help` to see available options. The included Slurm scripts `slurm/` provide examples for training on multiple GPUs. Please adjust their cluster settings before use.

Data preprocessing is available in [preprocessing/weirnet_preprocess.ipynb](preprocessing/weirnet_preprocess.ipynb).

## License

[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). See [LICENSE](LICENSE).
