include ../make.inc

all: libpangulu.so libpangulu.a


libpangulu.so:
	$(MPICPP) $(MPICPPFLAGS) -shared -fPIC -o $@ ../src/pangulu.o ../src/pangulu_cuda.o
libpangulu.a:
	ar -rv -o $@ ../src/pangulu.o ../src/pangulu_cuda.o
	- ranlib $@

clean:
	rm -f libpangulu.so
	rm -f libpangulu.a
	