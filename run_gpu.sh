#!/bin/bash
source venv/bin/activate

LIBGPUARRAY_PATH=~/tsabl/venv
CUDA_HOME=/usr/local/cuda

export CPATH=${CPATH}:${LIBGPUARRAY_PATH}/include:${CUDA_HOME}/include
export LIBRARY_PATH=${LIBRARY_PATH}:${LIBGPUARRAY_PATH}/lib:${CUDA_HOME}/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIBGPUARRAY_PATH}/lib:${CUDA_HOME}/lib64
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64

#KERAS_BACKEND=tensorflow
KERAS_BACKEND=theano

export THEANO_FLAGS="optimizer=fast_run,openmp=True,device=cuda,floatX=float32,nvcc.flags=-D_FORCE_INLINES"
export OMP_NUM_THREADS=2

python embeddings/main.py
