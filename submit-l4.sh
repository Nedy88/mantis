#!/bin/bash
#SBATCH --job-name=mantis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l4-24g:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --output=myjob.out
#SBATCH --error=myjob.out
./run.sh "$@"
