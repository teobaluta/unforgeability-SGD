CC = gcc
CFLAGS = -Wall -Wno-error -fpic  -std=c99 -O3 -fopenmp -I.
LDFLAGS = -shared  -fopenmp

RREF_LIB = librref.so
RREF_SRC_FILES = rref.c 

.PHONY: all clean

all: $(RREF_LIB)

$(RREF_LIB): $(RREF_SRC_FILES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(TARGET_LIB) $(RREF_LIB)
