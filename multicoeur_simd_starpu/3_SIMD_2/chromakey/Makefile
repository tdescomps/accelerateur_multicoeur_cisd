PRG = chromakey

CFLAGS  = -Wall -g -O3
CFLAGS += -std=gnu11
CFLAGS += -march=native -mavx2

.phony: all clean

all: $(PRG)

%.s: %.c
	$(CC) $(CFLAGS) $(CFLAGS_VEC) -S -o $@ $<

clean:
	rm -fv $(PRG) *.s

