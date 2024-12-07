PRGS=cholesky

CC=gcc

CFLAGS		= -g -Wall -O0

## BLAS / LAPACK
BLAS_CFLAGS	= -DUSE_MKL $(shell pkg-config --define-variable=MKLROOT=$MKLROOT --cflags mkl-dynamic-lp64-iomp)
BLAS_LIBS	= -DUSE_MKL $(shell pkg-config --define-variable=MKLROOT=$MKLROOT --libs mkl-dynamic-lp64-iomp)

CFLAGS		+= $(BLAS_CFLAGS)
LDLIBS		+= $(BLAS_LIBS)

%: %.c

%: %.o
	$(CC) -o $@ $< $(LDLIBS)
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

.phony: all clean
all: $(PRGS)
clean:
	rm -f $(PRGS)
