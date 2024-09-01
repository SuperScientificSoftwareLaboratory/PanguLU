LINK_METIS = /path/to/libmetis.a /path/to/libGKlib.a
OPENBLAS_LIB = /path/to/libopenblas.a
#LINK_CUDA = -L/path/to/cuda/lib64 -lcudart -lcusparse -lstdc++
LINK_PANGULU = ../lib/libpangulu.a # Derictly importing static library as compiler input makes dynamic library loader searching the directory of static library.

all: pangulu_example.elf

pangulu_example.elf:example.c
	mpicc -O3 $< -DCALCULATE_TYPE_R64 -I../include $(LINK_PANGULU) $(LINK_CUDA) $(LINK_METIS) $(OPENBLAS_LIB) -fopenmp -lpthread -lm -o $@

clean:
	rm -f *.elf
