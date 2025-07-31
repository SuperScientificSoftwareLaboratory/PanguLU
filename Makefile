all : examples

.PHONY : examples lib src clean update

examples : lib
	$(MAKE) -C $@

lib : src
	$(MAKE) -C $@

src: hunyuan
	$(MAKE) -C $@

hunyuan:
	$(MAKE) -C hunyuan_omp

clean:
	$(MAKE) -C src clean
	$(MAKE) -C lib clean
	$(MAKE) -C examples clean
	$(MAKE) -C hunyuan_omp clean

update : clean all