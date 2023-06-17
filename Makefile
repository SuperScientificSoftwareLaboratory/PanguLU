all:
	(cd src; make)
	(cd lib; make)
	(cd test; make)
src:
	(cd src; make)

lib:
	(cd lib; make)

test:
	(cd test; make)
clean:
	(cd src; make clean)
	(cd lib; make clean)
	(cd test; make clean)
