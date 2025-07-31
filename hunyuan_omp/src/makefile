CC = gcc
CFLAGS = -c -Wall -fPIC -fopenmp -O3

SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)

all: $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@
	mv $@ ../lib

clean:
	rm -f $(OBJECTS)