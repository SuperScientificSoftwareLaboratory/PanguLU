COMPILE_LEVEL = -O3

#0201000,GPU_CUDA
CUDA_PATH = /path/to/cuda
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart -lcusparse
NVCC = nvcc $(COMPILE_LEVEL)
NVCCFLAGS = $(PANGULU_FLAGS) -w -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 $(CUDA_INC) $(CUDA_LIB)

#general
CC = gcc $(COMPILE_LEVEL)
MPICC = mpicc $(COMPILE_LEVEL)
OPENBLAS_INC = -I/path/to/OpenBLAS/include
OPENBLAS_LIB = -L/path/to/OpenBLAS/lib -lopenblas
MPICCFLAGS = $(OPENBLAS_INC) $(CUDA_INC) $(OPENBLAS_LIB) -fopenmp -lpthread -lm
MPICCLINK = $(OPENBLAS_LIB)
METISFLAGS = -I/path/to/GKlib/include -I/path/to/METIS/include
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64 -DPANGULU_MC64 -DMETIS #-DGPU_OPEN -DHT_IS_OPEN
