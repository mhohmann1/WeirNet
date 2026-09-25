#!/bin/bash
#SBATCH --job-name=SuMoEvalPN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus-per-task=1
#SBATCH --partition=small_gpu
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=firstname.lastname@email.com
#SBATCH --output=slurmjob%j.log

module purge
module load miniforge3
module load cuda

eval "$(conda shell.bash hook)"
conda activate weirnet_env

export OMP_NUM_THREADS=1

python eval.py --model PN \
  --eval_data_path ./Data/PKW_OOD/combined_data.npz \
  --eval_pc_dir ./Data/PKW_OOD/pc \
  --eval_stl_dir ./Data/PKW_OOD/stl \
  --batch_size 32 --num_workers 6
