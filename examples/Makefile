LINK_METIS = /path/to/libparmetis.a /path/to/libmetis.a /path/to/libGKlib.a
OPENBLAS_LIB = /path/to/libopenblas.a
LINK_CUDA = -L/usr/local/cuda-12.2/lib64 -lcudart -lcusparse
LINK_PANGULU = ../lib/libpangulu.a # Derictly importing static library as compiler input makes dynamic library loader searching the directory of static library.

all: pangulu_driver.elf

pangulu_driver.elf:driver_r64.cpp
	mpic++ -O3 $< -I../include $(LINK_PANGULU) $(LINK_CUDA) $(LINK_METIS) $(OPENBLAS_LIB) -fopenmp -o $@

clean:
	rm -f *.elf *.tsv
