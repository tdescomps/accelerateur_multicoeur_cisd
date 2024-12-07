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

		printf(" %4.1f", array[i]);
	}
	printf(" ]");
}

/* Size in bytes of an SIMD register */
#define REG_BYTES (sizeof(__m256))

/* Number of elements in an SIMD register */
#define REG_NB_ELEMENTS (REG_BYTES / sizeof(float))

/* Compute the sum of vectors x and y, element by element, and store the result in vector z */
__attribute__((noinline)) void xpy (const float *x, const float *y, float *z, const int vector_size)
{
	/* Check that the vector size is a multiple of the number of elements in SIMD registers */
	assert(vector_size % REG_NB_ELEMENTS == 0);

	int i;
	for (i=0; i<vector_size; i+=REG_NB_ELEMENTS)
	{
		__m256 reg_x;
		__m256 reg_y;
		__m256 reg_z;

		/* Load A and B arrays in SIMD registers */
		reg_x = _mm256_load_ps(&x[i]);
		reg_y = _mm256_load_ps(&y[i]);

		/* Perform SIMD add operation */
		reg_z = _mm256_add_ps(reg_x, reg_y);

		/* Store SIMD register in C array */
		_mm256_store_ps(&z[i], reg_z);
	}
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

	float *z = aligned_alloc(REG_BYTES, nx*sizeof(*z));
	assert(z != NULL);
	memset(z, 0, nx*sizeof(*z));

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
	xpy(x, y, z, nx);

	struct timespec ts_begin;
	clock_gettime(CLOCK_MONOTONIC, &ts_begin);
	int i;
	for (i=0; i<nb_loops; i++)
	{
		xpy(x, y, z, nx);
	}
	struct timespec ts_end;
	clock_gettime(CLOCK_MONOTONIC, &ts_end);
	double timing = (ts_end.tv_sec - ts_begin.tv_sec) + 1.0e-9*(ts_end.tv_nsec - ts_begin.tv_nsec);

	if (verbose)
	{
		printf("z: ");
		disp_array(z, nx);
		printf("\n");
	}

	if (header)
	{
		printf("nx,seconds,gflops/s\n");
	}

	{
		double seconds = timing / nb_loops;
		double nbops = nb_loops*(double)nx;
		double flops = 1e-9 * nbops / timing;

		printf("%d,%.3le,%.3le\n", nx, seconds, flops);
	}

	free(x);
	free(y);
	free(z);

	return 0;
}

