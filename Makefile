all:
	(cd src; make)
	(cd lib; make)
	(cd examples; make)
src:
	(cd src; make)

lib:
	(cd lib; make)

examples:
	(cd examples; make)
clean:
	(cd src; make clean)
	(cd lib; make clean)
	(cd examples; make clean)

update:clean all