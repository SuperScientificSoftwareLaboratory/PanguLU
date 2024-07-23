include ../make.inc

all:pangulu.o pangulu_platforms

pangulu.o:pangulu.cpp
	$(MPICPP) $(MPICPPFLAGS) $(METISFLAGS) $(PANGULU_FLAGS) -c $< -o $@ -fPIC
	mv ./*.o ../lib

pangulu_platforms:
	cd .. && python3 build_helper.py compile_platform_code

clean:
	rm -f ../lib/pangulu*.o
