#!/bin/bash
#SBATCH --job-name=SuMo
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=8        
#SBATCH --partition=small_gpu8
#SBATCH --time=23:45:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=michael.hohmann@hsu-hh.de
#SBATCH --output=slurmjob%j.log

module purge
module load miniforge3
module load cuda

eval "$(conda shell.bash hook)"
conda activate weirnet_env

export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --nproc_per_node=8 --nnodes=1 \
  trainGNN.py --model GNN --epochs 500 --num_workers 6 --batch_size 4 --resume\