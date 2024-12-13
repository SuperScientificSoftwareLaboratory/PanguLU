COMPILE_LEVEL = -O3

#0201000,GPU_CUDA
CUDA_PATH = /usr/local/cuda
CUDA_INC = -I/path/to/cuda/include
CUDA_LIB = -L/path/to/cuda/lib64 -lcudart -lcusparse
NVCC = nvcc $(COMPILE_LEVEL)
NVCCFLAGS = $(PANGULU_FLAGS) -w -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 $(CUDA_INC) $(CUDA_LIB)

#general
CC = gcc $(COMPILE_LEVEL) #-fsanitize=address
MPICC = mpicc $(COMPILE_LEVEL) #-fsanitize=address
OPENBLAS_INC = -I/path/to/openblas/include
OPENBLAS_LIB = -L/path/to/openblas/lib -lopenblas
MPICCFLAGS = $(OPENBLAS_INC) $(CUDA_INC) $(OPENBLAS_LIB) -fopenmp -lpthread -lm
MPICCLINK = $(OPENBLAS_LIB)
METISFLAGS = -I/path/to/gklib/include -I/path/to/metis/include
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64 -DMETIS -DPANGULU_MC64 #-DGPU_OPEN -DHT_IS_OPEN
