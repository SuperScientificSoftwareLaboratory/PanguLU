include ../make.inc

all : oclean

libs : libpangulu.so libpangulu.a

libpangulu.so:
	$(MPICC) $(MPICCFLAGS) -shared -fPIC -o $@ ./pangulu*.o
libpangulu.a:
	ar -rv -o $@ ./pangulu*.o
	- ranlib $@

oclean: libs
	rm -f pangulu*.o

clean:
	rm -f libpangulu.so
	rm -f libpangulu.a
	