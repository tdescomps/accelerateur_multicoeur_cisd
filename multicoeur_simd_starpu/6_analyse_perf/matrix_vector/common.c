#include <stdlib.h>
#include <stdio.h>

#define DEFAULT_X_LEN 100
#define DEFAULT_y_LEN 100

#if ((defined MATRIX_VECTOR) && (defined MATRIX_T_VECTOR)) || ((!defined MATRIX_VECTOR) && (!defined MATRIX_T_VECTOR))
#error Either MATRIX_VECTOR or MATRIX_T_VECTOR must be defined
#endif

#ifdef ALIGN
#define ASSUME_ALIGNED(v) __assume_aligned(v, ALIGN)
#define ASSUME(e) __assume((e))
#else
#define ASSUME_ALIGNED(v)
#define ASSUME(e)
#endif

void vector_init(double *vector, const int len)
{
	int i;
	for (i=0; i<len; i++)
	{
		vector[i] = (double)(rand() % 100);
	}
}

void vector_zero(double *vector, const int len)
{
	int i;
	for (i=0; i<len; i++)
	{
		vector[i] = 0;
	}
}

void matrix_init(double *matrix, const int len_x, const int len_y)
{
	int j;
	for (j=0; j<len_x; j++)
	{
		int i;
		for (i=0; i<len_x; i++)
		{
			matrix[j*len_x + i] = (double)(rand() % 100);
		}
	}
}

__attribute__((noinline)) void vector_disp(double *vector, int len)
{
#ifdef DISP
	int i;
	for (i=0; i<len; i++)
	{
		printf("[%d]: %.3lf\n", i, vector[i]);
	}
#endif
}

__attribute__((noinline)) void matrix_disp(const double *matrix, const int len_x, const int len_y)
{
#ifdef DISP
	int j;
	printf("[[\n");
	for (j=0; j<len_y; j++)
	{
		printf("  j=%d:", j);
		int i;
		for (i=0; i<len_x; i++)
		{
			if (i>0)
			{
				printf(",");
			}
			printf(" %.3lf", matrix[j*len_x + i]);
		}
		printf("\n");
	}
	printf("]]\n");
#endif
}

#ifdef MATRIX_VECTOR
static __attribute__((noinline)) void matrix_vector(const double * restrict matrix, const double * restrict vector, const int len_x, const int len_y, double * restrict result_vector)
{
	ASSUME_ALIGNED(matrix);
	ASSUME_ALIGNED(vector);
	ASSUME_ALIGNED(result_vector);
	ASSUME(len_x%64 == 0);
	ASSUME(len_y%64 == 0);
	int j;
	for (j=0; j<len_y; j++)
	{
		double v = result_vector[j];
		int i;
		for (i=0; i<len_x; i++)
		{
			v += matrix[j*len_x + i] * vector[i];
		}
		result_vector[j] = v;
	}
}
#else
static __attribute__((noinline)) void matrix_t_vector(const double *matrix, const double *vector, const int len_x, const int len_y, double *result_vector)
{
	ASSUME_ALIGNED(matrix);
	ASSUME_ALIGNED(vector);
	ASSUME_ALIGNED(result_vector);
	ASSUME(len_x%64 == 0);
	ASSUME(len_y%64 == 0);
	int j;
	for (j=0; j<len_y; j++)
	{
		double v = result_vector[j];
		int i;
		for (i=0; i<len_x; i++)
		{
			v += matrix[i*len_y + j] * vector[i];
		}
		result_vector[j] = v;
	}
}
#endif

int main(int argc, char *argv[])
{
	double *vector = NULL;
	double *matrix = NULL;
	double *result_vector = NULL;
	int matrix_len_x = DEFAULT_X_LEN;
	int matrix_len_y = DEFAULT_X_LEN;
	
	if (argc > 1)
	{
		matrix_len_x = atoi(argv[1]);
		if (matrix_len_x < 1)
		{
			fprintf(stderr, "invalid matrix X length %d\n", matrix_len_x);
			exit(EXIT_FAILURE);
		}
		if (argc > 2)
		{
			matrix_len_y = atoi(argv[2]);
			if (matrix_len_y < 1)
			{
				fprintf(stderr, "invalid matrix Y length %d\n", matrix_len_y);
				exit(EXIT_FAILURE);
			}
			else
			{
				matrix_len_y = matrix_len_x;
			}
		}
	}

	printf("Matrix 'x' len: %d columns\n", matrix_len_x);
	printf("Matrix 'y' len: %d rows\n", matrix_len_y);

#ifdef MATRIX_VECTOR
	int vector_len = matrix_len_x;
	int result_vector_len = matrix_len_y;
#else
	int vector_len = matrix_len_y;
	int result_vector_len = matrix_len_x;
#endif

	printf("Vector len: %d elements\n", vector_len);
	printf("Result vector len: %d elements\n", result_vector_len);

#ifdef ALIGN
	vector = aligned_alloc(ALIGN, vector_len * sizeof(vector[0]));
#else
	vector = malloc(vector_len * sizeof(vector[0]));
#endif
	if (vector == NULL)
	{
		fprintf(stderr, "memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

#ifdef ALIGN
	result_vector = aligned_alloc(ALIGN, result_vector_len * sizeof(result_vector[0]));
#else
	result_vector = malloc(result_vector_len * sizeof(result_vector[0]));
#endif
	if (result_vector == NULL)
	{
		fprintf(stderr, "memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

#ifdef ALIGN
	matrix = aligned_alloc(ALIGN, matrix_len_x * matrix_len_y * sizeof(matrix[0]));
#else
	matrix = malloc(matrix_len_x * matrix_len_y * sizeof(matrix[0]));
#endif
	if (matrix == NULL)
	{
		fprintf(stderr, "memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	matrix_init(matrix, matrix_len_x, matrix_len_y);
	vector_init(vector, vector_len);

#ifdef DISP
	printf("Initial matrix':\n");
	matrix_disp(matrix, matrix_len_x, matrix_len_y);
	printf("\n");

	printf("Initial vector':\n");
	vector_disp(vector, vector_len);
	printf("\n");
#endif

	vector_zero(result_vector, result_vector_len);

#ifdef MATRIX_VECTOR
	matrix_vector(matrix, vector, matrix_len_x, matrix_len_y, result_vector);
#ifdef DISP
	printf("result of matrix x vector product:\n");
#endif
#else
	matrix_t_vector(matrix, vector, matrix_len_x, matrix_len_y, result_vector);
#ifdef DISP
	printf("result of matrix^t x vector product:\n");
#endif
#endif

	vector_disp(result_vector, result_vector_len);

#ifdef DISP
	printf("\n");
#endif

	free(vector);
	free(matrix);
	free(result_vector);

	return 0;
}
