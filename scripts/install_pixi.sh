#!/bin/bash
ROOT_DIR=/scratch/nedyalko_prisadnikov/tools/pixi
TOOLS_DIR=`dirname $ROOT_DIR`

if [ -d "$ROOT_DIR" ]; then
    echo "Pixi is already installed in $ROOT_DIR"
else
    echo "Installing Pixi in $ROOT_DIR..."
    mkdir -p $TOOLS_DIR
    curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=$ROOT_DIR TMP_DIR=$TOOLS_DIR/tmp bash
fi
