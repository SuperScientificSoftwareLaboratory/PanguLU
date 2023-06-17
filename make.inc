CUDA_PATH = /usr/local/cuda
LU_INC =-I../include
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart -lcusparse

NVCC = $(CUDA_PATH)/bin/nvcc
CPP = g++
MPICPP = mpic++

MPICPPFLAGS = -O3 -m64 -Wall -std=c++11 -lm -lpthread -fopenmp $(LU_INC)
NVCCFLAGS = -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 $(LU_INC) $(CUDA_INC) $(CUDA_LIB)
MPICPPLINK = -L/mpich/lib -lmpi

METISFLAGS = -lmetis
