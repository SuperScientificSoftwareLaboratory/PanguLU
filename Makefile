all : examples

.PHONY : examples lib src clean update

examples : lib
	$(MAKE) -C $@

lib : src
	$(MAKE) -C $@

src:
	$(MAKE) -C $@

clean:
	$(MAKE) -C src clean
	$(MAKE) -C lib clean
	$(MAKE) -C examples clean

update : clean all