PRG = matrix_vector matrix_t_vector

CFLAGS  = -g -Wall -std=c11
CFLAGS += -O3 -march=native
#CFLAGS += -fopt-info
#CFLAGS += -qopt-report -qopt-report-annotate=html
#CFLAGS += -DALIGN=64 -qopt-report -qopt-report-annotate=html

.phony: all clean

all: $(PRG)

matrix_vector: matrix_vector.c
	$(CC) $(CFLAGS) -DMATRIX_VECTOR -o $@ $< $(LDLIBS)
matrix_t_vector: matrix_t_vector.c
	$(CC) $(CFLAGS) -DMATRIX_T_VECTOR -o $@ $< $(LDLIBS)

clean:
	rm -fv $(PRG) *.o *.html *.optrpt

