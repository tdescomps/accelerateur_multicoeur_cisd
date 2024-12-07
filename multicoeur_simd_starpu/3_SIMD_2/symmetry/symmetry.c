#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define MAX_DISP_ARRAY 96
#define MAX_DISP_ROW_LEN 8

struct s_point
{
	float x;
	float y;
};

void usage (void)
{
	fprintf(stderr, "symmetry [--nb_points <NUMBER OF POINTS>]\n");
	exit(EXIT_FAILURE);
}

float simple_random_float(void)
{
	return ((float)((rand() % 40)-20)) / 10.0;
}

void fill_array(struct s_point *point_array, int len)
{
	int i;
	for (i=0; i<len; i++)
	{
		point_array[i].x = simple_random_float();
		point_array[i].y = simple_random_float();
	}
}

void disp_array(struct s_point *point_array, int len)
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
		else if (i % MAX_DISP_ROW_LEN == 0)
		{
			printf("\n");
			printf(" ");
		}

		if (i % MAX_DISP_ROW_LEN == 0)
		{
			printf(" %02d:", i);
		}
		printf(" (x:%4.1f, y:%4.1f)", point_array[i].x, point_array[i].y);
	}
	printf(" ]");
}

/* Size in bytes of an SIMD register */
#define REG_BYTES (sizeof(__m256))

/* Number of elements in an SIMD register */
#define REG_NB_ELEMENTS (REG_BYTES / sizeof(float))

__attribute__((noinline)) void symmetry_x (struct s_point *point_array, const int nb_points)
{
	/* Check that the number of points (two coordinates per point) is a
	 * multiple of the number of elements in SIMD registers */
	assert(nb_points % (REG_NB_ELEMENTS/2) == 0);

	__m256i x_vector;
	for (int i = 0; i < nb_points; i+= nb_points / REG_NB_ELEMENTS) {
		x_vector = _mm256_load_epi32(point_array + i);
		
	}
}

__attribute__((noinline)) void symmetry_y (struct s_point *point_array, const int nb_points)
{
	/* Check that the number of points (two coordinates per point) is a
	 * multiple of the number of elements in SIMD registers */
	assert(nb_points % (REG_NB_ELEMENTS/2) == 0);

	/*** ------------------------------------ ***/
	/*** add the function implementation here ***/
	/*** ------------------------------------ ***/
}

int main(int argc, char *argv[])
{
	int nb_points = 16;

	{
		int i = 1;
		while (i < argc)
		{
			char *arg = argv[i];
			if (strcmp(arg, "--nb_points") == 0)
			{
				i++;
				nb_points = atoi(argv[i]);
				if (nb_points < 1)
				{
					fprintf(stderr, "invalid nb_points: %d\n", nb_points);
					exit(EXIT_FAILURE);
				}
				if (nb_points % REG_NB_ELEMENTS != 0)
				{
					fprintf(stderr, "nb_points must be a multiple of the register size: %lu\n", REG_NB_ELEMENTS);
				}
			}
			else
			{
				usage();
				exit(EXIT_FAILURE);
			}
			i++;
		}
	}

	struct s_point *point_array = aligned_alloc(REG_BYTES, nb_points*sizeof(*point_array));
	assert(point_array != NULL);
	fill_array(point_array, nb_points);

	printf("Initial point array:\n");
	disp_array(point_array, nb_points);
	printf("\n");
	printf("\n");

	/* apply axis symmetry on x coordinates */
	symmetry_x(point_array, nb_points);

	printf("Point array after axis symmetry on x coordinates:\n");
	disp_array(point_array, nb_points);
	printf("\n");
	printf("\n");

	/* apply axis symmetry on y coordinates */
	symmetry_y(point_array, nb_points);

	printf("Point array after axis symmetry on y coordinates:\n");
	disp_array(point_array, nb_points);
	printf("\n");
	printf("\n");

	free(point_array);

	return 0;
}

