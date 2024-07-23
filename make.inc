COMPILE_LEVEL = -O3

#0201000,GPU_CUDA
CUDA_PATH = /usr/local/cuda-12.2
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart -lcusparse
NVCC = nvcc $(COMPILE_LEVEL)
NVCCFLAGS = $(PANGULU_FLAGS) -w -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 $(LU_INC) $(CUDA_INC) $(CUDA_LIB)


#general
CC = gcc $(COMPILE_LEVEL)
CPP = g++ $(COMPILE_LEVEL)
MPICPP = mpic++ $(COMPILE_LEVEL)

OPENBLAS_INC=-I/path/to/OpenBLAS/include
OPENBLAS_LIB=-L/path/to/OpenBLAS/lib -lopenblas

CPPFLAGS = -fopenmp
MPICPPFLAGS = -Wall -std=c++11 -lm -lpthread -fopenmp  $(LU_INC) $(OPENBLAS_INC) $(OPENBLAS_LIB) $(CUDA_INC)
MPICPPLINK = $(OPENBLAS_LIB)

METIS_INC = -I/path/to/metis/i64/include 
METIS_LIB = -L/path/to/metis/i64/lib 

PARMETIS_INC = -I/path/to/parmetis/i64/include
PARMETIS_LIB = -L/path/to/parmetis/i64/lib 

METISFLAGS = $(METIS_INC) $(METIS_LIB) $(PARMETIS_INC) $(PARMETIS_LIB) -I/path/to/GKlib/include /path/to/libGKlib.a -I/path/to/metis/i64/include /path/to/metis/i64/lib/libmetis.a -I/path/to/parmetis/i64/include /path/to/libparmetis.a

PANGULU_FLAGS = -DCALCULATE_TYPE_R64 -DGPU_OPEN #-DHT_IS_OPEN
