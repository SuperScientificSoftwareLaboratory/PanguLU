include ../make.inc

all: libpangulu.so


libpangulu.so:
	$(MPICPP) $(MPICPPFLAGS) -shared -fPIC -o $@ ../src/pangulu.o ../src/pangulu_cuda.o
	cp libpangulu.so ../test

clean:
	rm -f libpangulu.so
	rm -f ../test/libpangulu.so
