include ../../../../../make.inc
pangulu_0201000.o:pangulu_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(METIS_INC) -Xcompiler -fPIC  -c $< -o $@
	mv $@ ../../../../../lib