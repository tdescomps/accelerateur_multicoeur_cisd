#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	double *a = malloc(4 * sizeof(double));
	double *b = malloc(4 * sizeof(double));
	__m128d m_a = _mm256_load_ps
	_mm_store_sd(a, m_a);

	printf("a[0]: %lf\n", a[0]);
	printf("a[1]: %lf\n", a[1]);
	return EXIT_SUCCESS;
}