all : examples

.PHONY : examples lib src clean update

examples : lib
	$(MAKE) -C $@

lib : src
	$(MAKE) -C $@

src: reordering
	$(MAKE) -C $@

reordering:
	$(MAKE) -C reordering_omp

clean:
	$(MAKE) -C src clean
	$(MAKE) -C lib clean
	$(MAKE) -C examples clean
	$(MAKE) -C reordering_omp clean

update : clean all