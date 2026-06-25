LINK_METIS = $(PATH_TO_METIS_I32_LIB)/libmetis.so
LINK_OPENBLAS = $(PATH_TO_OPENBLAS_LIB)/libopenblas.so
LINK_CUDA = -L$(PATH_TO_CUDA_LIB) -lcudart -lstdc++ -lcusolver -lcublas
LINK_PANGULU = ../lib/libpangulu.a

all: pangulu_example.elf

pangulu_example.elf:example.c
	mpicc -O3 $< -DCALCULATE_TYPE_R64 -I../include $(LINK_PANGULU) $(LINK_CUDA) $(LINK_REORDERING) $(LINK_METIS) $(LINK_OPENBLAS)  -fopenmp -lpthread -lm -o $@

clean:
	rm -f *.elf
