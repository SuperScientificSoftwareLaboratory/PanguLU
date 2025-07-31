all: lib

lib: src
	$(MAKE) -C lib

src: generate_mynd_types
	$(MAKE) -C src

generate_mynd_types: clean
	echo "" > include/mynd_types.h
	echo "#define IDXTYPEWIDTH 32" >> include/mynd_types.h
	echo "#define REALTYPEWIDTH 64" >> include/mynd_types.h

clean:
	$(MAKE) -C lib clean
	- (rm include/mynd_types.h)

