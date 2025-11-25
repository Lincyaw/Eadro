#!/bin/bash -ex
export ALGORITHM=${ALGORITHM:-eadro}
export CHECKPOINT_PATH=/home/nn/workspace/Eadro/result/37dee8ed/model.ckpt
LOGURU_COLORIZE=0 infer/bin/python main.py container run