#!/usr/bin/env bash
source venv/bin/activate

# Set to theano or tensorflow
export KERAS_BACKEND=theano

# If using theano, specify paths to libgpuarray and CUDA
LIBGPUARRAY_PATH=~/tsabl/venv
CUDA_HOME=/usr/local/cuda

# Should not be changed
export CPATH=${CPATH}:${LIBGPUARRAY_PATH}/include:${CUDA_HOME}/include
export LIBRARY_PATH=${LIBRARY_PATH}:${LIBGPUARRAY_PATH}/lib:${CUDA_HOME}/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIBGPUARRAY_PATH}/lib:${CUDA_HOME}/lib64

export THEANO_FLAGS=device=cuda,floatX=float32,nvcc.flags=-D_FORCE_INLINES

python embeddings/main.py
