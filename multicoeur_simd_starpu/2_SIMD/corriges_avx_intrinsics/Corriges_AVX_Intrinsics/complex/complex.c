#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define MAX_DISP_ARRAY 10

#define REG_BYTES (sizeof(__m256i))
#define REG_ELEMENT_TYPE int32_t
#define REG_ELEMENT_TYPE_STR "int32_t"
#define REG_NB_ELEMENTS (REG_BYTES / sizeof(REG_ELEMENT_TYPE))

void print_format(void)
{
	printf("|");
	int i;
	for (i=0; i<REG_NB_ELEMENTS; i++)
	{
		if (i > 0)
		{
			if (i % (REG_NB_ELEMENTS / 2) == 0)
			{
				printf("   ::  ");
			}
			else if (i % (REG_NB_ELEMENTS / 4) == 0)
			{
				printf("  : ");
			}
			else
			{
				printf(" .");
			}
		}
		printf(" %lu", REG_NB_ELEMENTS-i-1);
	}

	printf(" |");
}
void print_reg(__m256i r)
{
	uint32_t ri[REG_NB_ELEMENTS];
	_mm256_storeu_si256((__m256i *)ri, r);
	printf("|");
	int i;
	for (i=0; i<REG_NB_ELEMENTS; i++)
	{
		if (i > 0)
		{
			if (i % (REG_NB_ELEMENTS / 2) == 0)
			{
				printf("   ::  ");
			}
			else if (i % (REG_NB_ELEMENTS / 4) == 0)
			{
				printf("  : ");
			}
			else
			{
				printf(" .");
			}
		}
		if ((REG_NB_ELEMENTS-i-1) % 2 == 1)
		{
			printf("< ");
		}
		printf(" %3d", ri[REG_NB_ELEMENTS-i-1]);
		if ((REG_NB_ELEMENTS-i-1) % 2 == 1)
		{
			printf("i");
		}
		else
		{
			printf(" >");
		}
	}

	printf(" |");
}

void print_half_reg(__m128i r)
{
	uint32_t ri[REG_NB_ELEMENTS];
	_mm_store_si128((__m128i *)ri, r);
	printf("|");
	int i;
	for (i=0; i<REG_NB_ELEMENTS/2; i++)
	{
		if (i > 0)
		{
			if (i % (REG_NB_ELEMENTS / 4) == 0)
			{
				printf("  : ");
			}
			else
			{
				printf(" .");
			}
		}
		printf(" %4d", ri[REG_NB_ELEMENTS/2-i-1]);
	}

	printf(" |");
}

void print_half_reg_hex(__m128i r)
{
	uint32_t ri[REG_NB_ELEMENTS];
	_mm_store_si128((__m128i *)ri, r);
	printf("|");
	int i;
	for (i=0; i<REG_NB_ELEMENTS/2; i++)
	{
		if (i > 0)
		{
			if (i % (REG_NB_ELEMENTS / 4) == 0)
			{
				printf("  : ");
			}
			else
			{
				printf(" .");
			}
		}
		printf(" 0x%08x", ri[REG_NB_ELEMENTS/2-i-1]);
	}

	printf(" |");
}

void print_reg_hex(__m256i r)
{
	uint32_t ri[REG_NB_ELEMENTS];
	_mm256_storeu_si256((__m256i *)ri, r);
	printf("|");
	int i;
	for (i=0; i<REG_NB_ELEMENTS; i++)
	{
		if (i > 0)
		{
			if (i % (REG_NB_ELEMENTS / 2) == 0)
			{
				printf("   ::  ");
			}
			else if (i % (REG_NB_ELEMENTS / 4) == 0)
			{
				printf("  : ");
			}
			else
			{
				printf(" .");
			}
		}
		printf(" 0x%08x", ri[REG_NB_ELEMENTS-i-1]);
	}

	printf(" |");
}

void print_array(int32_t v[8])
{
	int i;
	for (i=0; i<8; i++)
	{
		if (i>0)
		{
			printf(",");
		}
		printf(" %4d", v[i]);
	}
}

void print_array_reverse(int32_t v[8])
{
	int i;
	for (i=0; i<8; i++)
	{
		if (i>0)
		{
			printf(",");
		}
		printf(" %4d", v[7-i]);
	}
}

void print_data(int32_t *data, int n)
{
	int i;
	for (i=0; i<n; i++)
	{
		if (i>0)
		{
			printf(",");
		}
		printf(" %2d", data[i]);
	}
}

__attribute__((noinline)) void set0(__m256i *x)
{
	*x = _mm256_setzero_si256();
}

__attribute__((noinline)) void set(__m256i *x, int32_t v[8])
{
	*x = _mm256_set_epi32(v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0]);
}

__attribute__((noinline)) void cmplx_sum(__m256i x, __m256i y, __m256i *z)
{
	*z = _mm256_add_epi32(x, y);
}

__attribute__((noinline)) void cmplx_prod(__m256i x, __m256i y, __m256i *z)
{
	__m256i factor = _mm256_set_epi32(-1, 1, -1, 1, -1, 1, -1, 1);
	__m256i real_mask = _mm256_set_epi32(0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff);

	__m256i real = _mm256_mullo_epi32(x, y);
	real = _mm256_mullo_epi32(factor, real);
	__m256i real_shuf = _mm256_shuffle_epi32(real, _MM_SHUFFLE(2, 3, 0, 1));
	real = _mm256_add_epi32(real, real_shuf);
	real = _mm256_and_si256(real_mask, real);

	__m256i y_shuf =  _mm256_shuffle_epi32(y, _MM_SHUFFLE(2, 3, 0, 1));
	__m256i imag = _mm256_mullo_epi32(x, y_shuf);
	__m256i imag_shuf =  _mm256_shuffle_epi32(imag, _MM_SHUFFLE(2, 3, 0, 1));
	imag = _mm256_add_epi32(imag, imag_shuf);
	imag = _mm256_andnot_si256(real_mask, imag);

	*z = _mm256_or_si256(real, imag);
}


int main(int argc, char *argv[])
{
	{
		int32_t vx[8] = {3, 8, 3, 2, 6, 4, 1, 3};
		int32_t vy[8] = {4, 2, 3, 2, 6, 1, 5, 7};

		__m256i x;
		__m256i y;
		__m256i z;

		set(&x, vx);
		set(&y, vy);
		set0(&z);

		printf("cmplx_sum( ");
		print_reg(x);
		printf(",\n");
		printf("           ");
		print_reg(y);
		printf("):\n\n");

		cmplx_sum(x, y, &z);
		printf("-->        ");
		print_reg(z);
		printf("\n\n");
	}

	{
		int32_t vx[8] = {3, 8, 3, 2, 6, 4, 1, 3};
		int32_t vy[8] = {4, 2, 3, 2, 6, 1, 5, 7};

		__m256i x;
		__m256i y;
		__m256i z;

		set(&x, vx);
		set(&y, vy);
		set0(&z);

		printf("cmplx_prod(");
		print_reg(x);
		printf(",\n");
		printf("           ");
		print_reg(y);
		printf("):\n\n");

		cmplx_prod(x, y, &z);
		printf("-->        ");
		print_reg(z);
		printf("\n\n");
	}

	return 0;
}

