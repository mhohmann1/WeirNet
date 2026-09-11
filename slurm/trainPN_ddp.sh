#!/bin/bash
#SBATCH --job-name=SuMo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=2
#SBATCH --partition=small_gpu
#SBATCH --time=23:45:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=firstname.lastname@email.com
#SBATCH --output=slurmjob%j.log

module purge
module load miniforge3
module load cuda

eval "$(conda shell.bash hook)"
conda activate weirnet_env

echo $CUDA_VISIBLE_DEVICES

export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=2 --nnodes=1 \
  train_ddp.py --model PN --epochs 500 --n_points 5000 --num_workers 6 --resume \