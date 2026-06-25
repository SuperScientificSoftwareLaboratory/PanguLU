include ../../../../../make.inc
pangulu_platform_0201000.o:pangulu_platform_0201000.cu
	$(NVCC) $(COMPILE_LEVEL) $(NVCCFLAGS) $(PANGULU_FLAGS) -Xcompiler -fPIC  -c $< -o $@
	mv $@ ../../../../../lib