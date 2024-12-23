#!/bin/bash
bash scripts/install_pixi.sh
. ~/.bashrc
cd /home/nedyalko_prisadnikov/mantis
echo "Running pixi install."
pixi install
pixi run torchrun --nnodes=1 --nproc-per-node=$NUM_GPUS src/train.py "$@"
