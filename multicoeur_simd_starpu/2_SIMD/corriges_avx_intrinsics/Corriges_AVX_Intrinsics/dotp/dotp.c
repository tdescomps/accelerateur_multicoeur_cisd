#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define MAX_DISP_ARRAY 64

void usage (void)
{
	fprintf(stderr, "axpy [--nx <VECTOR 'x' SIZE>] [--nb-loops <number of loops>] [--header] [--verbose]\n");
	exit(EXIT_FAILURE);
}

void fill_array(float *array, int len)
{
	int i;
	for (i=0; i<len; i++)
	{
		array[i] = ((float)((rand() % 40)-20)) / 10.0;
	}
}

void disp_array(float *array, int len)
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

		printf(" %2.1f", array[i]);
	}
	printf(" ]");
}

/* Size in bytes of an SIMD register */
#define REG_BYTES (sizeof(__m256))

/* Number of elements in an SIMD register */
#define REG_NB_ELEMENTS (REG_BYTES / sizeof(float))

/* Compute "alpha * x + y" , element by element, and store the result in vector z */
__attribute__((noinline)) float dot_product (const float *x, const float *y, const int vector_size)
{
	assert(vector_size % REG_NB_ELEMENTS == 0);
	float result = 0;

	if (vector_size > 0)
	{
		__m256 reg_acc = _mm256_setzero_ps();

		int i;
		for (i=0; i<vector_size; i+=REG_NB_ELEMENTS)
		{
			__m256 reg_x;
			__m256 reg_y;

			/* Load A and B arrays in SIMD registers */
			reg_x = _mm256_load_ps(&x[i]);
			reg_y = _mm256_load_ps(&y[i]);

			/* Perform SIMD add operation */
			reg_acc = _mm256_fmadd_ps(reg_x, reg_y, reg_acc);

		}

		/* Perform the sum reduction */
		reg_acc = _mm256_add_ps(reg_acc, _mm256_permute2f128_ps(reg_acc, reg_acc, _MM_SHUFFLE(0,0,0,1)));
		reg_acc = _mm256_add_ps(reg_acc, _mm256_permute_ps(reg_acc, _MM_SHUFFLE(1,0,3,2)));
		reg_acc = _mm256_add_ps(reg_acc, _mm256_permute_ps(reg_acc, _MM_SHUFFLE(2,3,0,1)));

		result = reg_acc[0];
	}
	return result;
}

int main(int argc, char *argv[])
{
	int nb_loops = 10;
	int nx = 16;
	int verbose = 0;
	int header = 0;

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
			else if (strcmp(arg, "--nb-loops") == 0)
			{
				i++;
				nb_loops = atoi(argv[i]);
				if (nb_loops < 1)
				{
					fprintf(stderr, "invalid number of loops: %d\n", nb_loops);
					exit(EXIT_FAILURE);
				}
			}
			else if (strcmp(arg, "--header") == 0)
			{
				header = 1;
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
	fill_array(x, nx);

	float *y = aligned_alloc(REG_BYTES, nx*sizeof(*y));
	assert(y != NULL);
	fill_array(y, nx);

	if (verbose)
	{
		printf("x: ");
		disp_array(x, nx);
		printf("\n");

		printf("y: ");
		disp_array(y, nx);
		printf("\n");
	}

	/* warmup iteration */
	float dp = dot_product(x, y, nx);

	struct timespec ts_begin;
	clock_gettime(CLOCK_MONOTONIC, &ts_begin);
	int i;
	for (i=0; i<nb_loops; i++)
	{
		dp = dot_product(x, y, nx);
	}
	struct timespec ts_end;
	clock_gettime(CLOCK_MONOTONIC, &ts_end);
	double timing = (ts_end.tv_sec - ts_begin.tv_sec) + 1.0e-9*(ts_end.tv_nsec - ts_begin.tv_nsec);

	if (verbose)
	{
		printf("dot product: %.2f\n", dp);
	}

	if (header)
	{
		printf("nx,seconds,gflops/s\n");
	}

	{
		double seconds = timing / nb_loops;
		double nbops = nb_loops*(double)nx;
		double flops = 2 * 1e-9 * nbops / timing; /* fma == 2 operations */

		printf("%d,%.3le,%.3le\n", nx, seconds, flops);
	}

	free(x);
	free(y);

	return 0;
}

