include ../make.inc



all:pangulu.o pangulu_cuda.o

pangulu.o:pangulu.cpp
	$(MPICPP) $(MPICPPFLAGS) $(METIS_INC) -c $< -o $@ -fPIC

pangulu_cuda.o:pangulu_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(METIS_INC) -Xcompiler -fPIC  -c $< -o $@ 




clean:
	rm -f pangulu.o pangulu_cuda.o



