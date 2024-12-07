#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <time.h>

#if defined USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#elif defined USE_PLASMA
#include <cblas.h>
#include <lapacke.h>
#else /* default OpenBLAS */
#include <cblas_openblas.h>
#include <lapacke.h>
#endif

#define _(row,col,ld) ((row)+(col)*(ld))

static void fill(double *A, int n, int lda) {
	int ret;
	int i;
	int seed[] = {0,0,0,1};

	ret = LAPACKE_dlarnv(1 /* uniform 0..1 */, seed, n*lda, A);
	assert(ret == 0);
	for (i=0; i<n; i++) {
		int j;
		A[_(i,i,lda)] += n;
		for (j=i+1; j<n; j++) {
			A[_(i,j,lda)] = 0;
		}
	}
}

static void ref_cholesky(double *refA, int n, int lda) {
	int ret = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, refA, lda);
	assert(ret == 0);
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		fprintf(stderr, "usage: %s <N>\n", argv[0]);
		exit(1);
	}
	int n = atoi(argv[1]);
	if (n < 1) {
		fprintf(stderr, "N must be >= 1\n");
		exit(1);
	}

	double *A = malloc(n*n*sizeof(*A));
	assert(A != NULL);

	printf("n=%d\n", n);

	printf("fill start\n");
	fill(A, n, n);
	printf("fill end\n");

	struct timespec ts_begin, ts_end;
	printf("computation start\n");
	clock_gettime(CLOCK_MONOTONIC, &ts_begin);
	ref_cholesky(A, n, n);
	clock_gettime(CLOCK_MONOTONIC, &ts_end);
	printf("computation end\n");
	double timing = (ts_end.tv_sec - ts_begin.tv_sec) + 1.0e-9*(ts_end.tv_nsec - ts_begin.tv_nsec);
	printf("computation time: %.2le s\n", timing);

	free(A);

	return 0;
}
