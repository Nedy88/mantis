#!/bin/bash
#SBATCH --job-name=mantis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --output=myjob.out
#SBATCH --error=myjob.out
export NUM_GPUS=2
./run.sh "$@"

