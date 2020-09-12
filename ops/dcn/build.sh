#!/usr/bin/env bash

# You may need to modify the following paths before compiling.
CUDA_HOME=/usr/local/cuda-10.1 \
CUDNN_INCLUDE_DIR=/usr/local/cuda-10.1/include \
CUDNN_LIB_DIR=/usr/local/cuda-10.1/lib64 \

python setup.py build_ext --inplace

if [ -d "build" ]; then
    rm -r build
fi
