#!/bin/bash
source venv/bin/activate

LIBGPUARRAY_PATH=~/tsabl/venv
CUDA_PATH=/usr/local/cuda

export CPATH=${CPATH}:${LIBGPUARRAY_PATH}/include:${CUDA_PATH}/include
export LIBRARY_PATH=${LIBRARY_PATH}:${LIBGPUARRAY_PATH}/lib:${CUDA_PATH}/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIBGPUARRAY_PATH}/lib:${CUDA_PATH}/lib64

THEANO_FLAGS=device=cuda,floatX=float32,nvcc.flags=-D_FORCE_INLINES python embeddings/main.py
:
