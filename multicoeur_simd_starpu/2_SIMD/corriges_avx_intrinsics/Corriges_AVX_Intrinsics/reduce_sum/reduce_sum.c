#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define MAX_DISP_ARRAY 32

void usage (void)
{
	fprintf(stderr, "xpy [--nx <VECTOR 'x' SIZE>] [--nb-loops <number of loops>] [--header] [--verbose]\n");
	exit(EXIT_FAILURE);
}

void vector_init(float *vector, int len)
{
	int i;
	for (i=0; i<len; i++)
	{
		vector[i] = (float)(rand() % 10);
	}
}

void vector_disp(float *vector, int len)
{
	int i;
	printf("[");
	for (i=0; i<len; i++)
	{
		if (i > 0)
		{
			printf(",");
		}

		if (i >= MAX_DISP_ARRAY)
		{
			printf(" ...");
			break;
		}

		printf(" %2.1f", vector[i]);
	}
	printf(" ]");
}

/* Size in bytes of an SIMD register */
#define REG_BYTES (sizeof(__m256))

/* Number of elements in an SIMD register */
#define REG_NB_ELEMENTS (REG_BYTES / sizeof(float))

void print_reg(__m256 r)
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
		printf(" %4.1f", r[REG_NB_ELEMENTS-i-1]);
	}

	printf(" |");
}

/* Compute the sum of vectors x and y, element by element, and store the result in vector z */
__attribute__((noinline)) void reduce_sum (const float *x, const int vector_size, __m256 *r)
{
	/* Check that the vector size is a multiple of the number of elements in SIMD registers */
	assert(vector_size % REG_NB_ELEMENTS == 0);

	__m256 a;
	if (vector_size > 0)
	{
		/* Initialize the accumulator with the first elements of x */
		a = _mm256_load_ps(&x[0]);

		int i;
		for (i=REG_NB_ELEMENTS; i<vector_size; i+=REG_NB_ELEMENTS)
		{
			/* Accumulate the remaining of the x in a */
			__m256 v = _mm256_load_ps(&x[i]);
			a = _mm256_add_ps(a, v);
		}

		/* Add elements in the lower 128-bit
		 * and the elements in the higher 128-bit
		 * This sums element i with element i + 4 */
		a = _mm256_add_ps(a, _mm256_permute2f128_ps(a, a, _MM_SHUFFLE(0,0,0,1)));

		/*  This sums element i with element i + 2 */
		a = _mm256_add_ps(a, _mm256_permute_ps(a, _MM_SHUFFLE(1,0,3,2)));

		/* This sums element i with element i + 1 */
		a = _mm256_add_ps(a, _mm256_permute_ps(a, _MM_SHUFFLE(2,3,0,1)));
	}
	else
	{
		a = _mm256_setzero_ps();
	}
	*r = a;

}

int main(int argc, char *argv[])
{
	int nx = 16;
	int verbose = 0;

	{
		int i = 1;
		while (i < argc)
		{
			char *arg = argv[i];
			if (strcmp(arg, "--nx") == 0)
			{
				i++;
				nx = atoi(argv[i]);
				if (nx < 1)
				{
					fprintf(stderr, "invalid nx: %d\n", nx);
					exit(EXIT_FAILURE);
				}
				if (nx % REG_NB_ELEMENTS != 0)
				{
					fprintf(stderr, "nx must be a multiple of the register size: %lu\n", REG_NB_ELEMENTS);
				}
			}
			else if (strcmp(arg, "--verbose") == 0)
			{
				verbose = 1;
			}
			else
			{
				usage();
				exit(EXIT_FAILURE);
			}
			i++;
		}
	}

	float *x = aligned_alloc(REG_BYTES, nx*sizeof(*x));
	assert(x != NULL);
	vector_init(x, nx);

	if (verbose)
	{
		printf("x: ");
		vector_disp(x, nx);
		printf("\n");

	}

	__m256 result;
	
	reduce_sum(x, nx, &result);
	print_reg(result);
	printf("\n\n");

	free(x);

	return 0;
}

