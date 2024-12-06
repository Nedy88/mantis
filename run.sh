#!/bin/bash
bash scripts/install_pixi.sh
source ~/.bashrc
cd /home/nedyalko_prisadnikov/mantis
pixi install
pixi run torchrun --nnodes=1 --nproc-per-node=8 src/train.py "$@"
