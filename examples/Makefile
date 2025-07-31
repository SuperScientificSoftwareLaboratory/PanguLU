LINK_HUNYUAN = ../hunyuan_omp/lib/libmynd.so
#LINK_METIS = /path/to/metis/lib/libmetis.so
LINK_OPENBLAS = /path/to/openblas/lib/libopenblas.so
LINK_CUDA = -L/path/to/cuda/lib64 -lcudart -lstdc++ -lcusolver -lcublas
LINK_PANGULU = ../lib/libpangulu.a

all: pangulu_example.elf

pangulu_example.elf:example.c
	mpicc -O3 $< -DCALCULATE_TYPE_R64 -I../include $(LINK_PANGULU) $(LINK_CUDA) $(LINK_HUNYUAN) $(LINK_METIS) $(LINK_OPENBLAS)  -fopenmp -lpthread -lm -o $@

clean:
	rm -f *.elf
