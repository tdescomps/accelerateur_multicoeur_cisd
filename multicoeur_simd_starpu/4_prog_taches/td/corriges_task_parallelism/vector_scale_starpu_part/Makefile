PRG = vector_scale_starpu_part

CFLAGS = -g -Wall

STARPU_VERSION = starpu-1.3
CFLAGS += $(shell pkg-config --cflags $(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs $(STARPU_VERSION))

.phony: all clean

all: $(PRG)
clean:
	rm -fv $(PRG)

