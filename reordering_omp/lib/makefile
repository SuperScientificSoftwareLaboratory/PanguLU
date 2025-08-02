CC = gcc
STATIC_LIB = libmynd.a
SHARED_LIB = libmynd.so
OBJECTS = $(wildcard *.o)

all: $(STATIC_LIB) $(SHARED_LIB)
	$(RM) $(OBJECTS) 

$(SHARED_LIB): $(OBJECTS)
	$(CC) -shared $(OBJECTS) -o $@

$(STATIC_LIB): $(OBJECTS)
	ar rcs $@ $(OBJECTS)

clean:
	rm -f $(SHARED_LIB) $(STATIC_LIB) *.o
