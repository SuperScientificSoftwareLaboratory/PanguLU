COMPILE_LEVEL = -O3
CUDA_COMPILE_LEVEL =

#0201000,GPU_CUDA
CUDA_INC = -I/path/to/cuda/include
NVCC = nvcc $(CUDA_COMPILE_LEVEL) $(COMPILE_LEVEL)
NVCCFLAGS = $(PANGULU_FLAGS) -w -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 $(CUDA_INC)

#general
CC = gcc $(COMPILE_LEVEL)
MPICC = mpicc $(COMPILE_LEVEL)
OPENBLAS_INC = -I/path/to/openblas/include
CFLAGS = $(OPENBLAS_INC) $(CUDA_INC) -fopenmp -lpthread -lm
MPICCFLAGS = $(CFLAGS)
HUNYUAN_INC = -I../hunyuan_omp/include
#METIS_INC = -I/path/to/metis/include
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64 -DGPU_OPEN #-DMETIS #-DPANGULU_MC64 #-DPANGULU_PERF
