include ../make.inc

EXE=PanguLU

all: $(EXE)


$(EXE):test.o
	$(MPICPP) test.o ../lib/libpangulu.a $(MPICPPLINK) -I../include -L. $(CUDA_INC) $(CUDA_LIB) $(METISFLAGS) -fopenmp -o $(EXE)


test.o:test.cpp
	$(MPICPP) $(MPICPPFLAGS) -c $< -o $@


clean:
	rm -f test.o $(EXE)

