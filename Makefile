all : examples

.PHONY : examples lib src clean update

examples : lib
	$(MAKE) -C $@

lib : src
	$(MAKE) -C $@

src:
	$(MAKE) -C $@

clean:
	(cd src; $(MAKE) clean)
	(cd lib; $(MAKE) clean)
	(cd examples; $(MAKE) clean)

update : clean all