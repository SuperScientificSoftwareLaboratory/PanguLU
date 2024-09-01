include ../make.inc
all:pangulu_host pangulu_platforms

src:=$(wildcard *.c)
pangulu_host:$(src:.c=.o) 

%.o:%.c
	$(MPICC) $(MPICCFLAGS) $(METISFLAGS) $(PANGULU_FLAGS) -c $< -o $@ -fPIC
	mv $@ ../lib

pangulu_platforms:
	cd .. && python3 build_helper.py compile_platform_code

clean:
	-(rm -f ../lib/pangulu*.o)
	-(rm -f ./pangulu*.o)
