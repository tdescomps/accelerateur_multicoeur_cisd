#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	double *a = malloc(2 * sizeof(double));
	a[0] = 100; a[1] = 101;
	__m128d m_a = _mm_set_pd(1.0, 2.0);
	_mm_store_sd(a, m_a);

	printf("a[0]: %lf\n", a[0]);
	printf("a[1]: %lf\n", a[1]);
	return EXIT_SUCCESS;
}