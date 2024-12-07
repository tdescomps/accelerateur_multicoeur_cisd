#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	double a[2]      = {1.0, 2.0};
	double b[2]      = {4.0, 2.0};
	double result[2] = {28., 28.};

	//__m128d m_a = _mm_load_sd(a);
	__m128d m_a = _mm_set_pd(1.0, 2.0);
	__m128d m_b = _mm_load_sd(b);

	__m128d m_result = _mm_cmpeq_sd(m_a, m_b);

	_mm_store_sd(a, m_a);
	_mm_store_sd(b, m_b);
	_mm_store_sd(result, m_result);

	printf("a[0]: %lf\n", a[0]);
	printf("a[1]: %lf\n", a[1]);
	printf("b[0]: %lf\n", b[0]);
	printf("b[1]: %lf\n", b[1]);
	printf("result[0]: %lf\n", result[0]);
	printf("result[1]: %lf\n", result[1]);

	return EXIT_SUCCESS;
}