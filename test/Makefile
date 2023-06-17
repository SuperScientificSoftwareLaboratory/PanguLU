include ../make.inc

EXE=PanguLU

all: $(EXE)


$(EXE):test.o
	$(NVCC) $(CUDA_INC) $(CUDA_LIB)  $(MPICPPLINK) test.o -I../include -L. -lpangulu $(METISFLAGS) -o $(EXE)


test.o:test.cpp
	$(MPICPP) $(MPICPPFLAGS) -c $< -o $@


clean:
	rm -f test.o $(EXE)

