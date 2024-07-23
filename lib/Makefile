include ../make.inc

all: libpangulu.so libpangulu.a oclean


libpangulu.so:
	$(MPICPP) $(MPICPPFLAGS) -shared -fPIC -o $@ ./pangulu*.o
libpangulu.a:
	ar -rv -o $@ ./pangulu*.o
	- ranlib $@

oclean:
	rm -f pangulu*.o

clean:
	rm -f libpangulu.so
	rm -f libpangulu.a
	