PRGS = mandelbrot_x11

CFLAGS  = -Wall -g -O3
CFLAGS += -std=gnu11
CFLAGS += -march=native -mavx2

LDLIBS=-lX11

.PHONY: all clean

all: $(PRGS)

clean:
	rm -vf *.o $(PRGS)

