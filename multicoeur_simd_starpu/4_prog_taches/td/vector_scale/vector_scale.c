#include <stdlib.h>
#include <stdio.h>
#include <starpu.h>

#define DEFAULT_VECTOR_X_LEN 100

void vector_init(double *vector, int len)
{
	int i;
	for (i = 0; i < len; i++)
	{
		vector[i] = (double)(rand() % 100);
	}
}

void vector_disp(double *vector, int len)
{
	int i;
	for (i = 0; i < len; i++)
	{
		printf("[%d]: %.3lf\n", i, vector[i]);
	}
}

void vector_scale(double *vector, int len, double factor)
{
	int i;
	for (i = 0; i < len; i++)
	{
		vector[i] *= factor;
	}
}

void vector_scale_func(void *buffers[], void *cl_arg)
{
	struct starpu_vector_interface *vector_handle = buffers[0];
	unsigned n = STARPU_VECTOR_GET_NX(vector_handle);
	double *vector = (double*) STARPU_VECTOR_GET_PTR(vector_handle);
	double factor;
	starpu_codelet_unpack_args(cl_arg, &factor);

	vector_scale(vector, n, factor);
}

struct starpu_codelet mult_codelet = {
	.cpu_funcs = {vector_scale_func},
	.nbuffers = 1,
	.modes = {STARPU_RW},
};

int main(int argc, char *argv[])
{
	int ret = starpu_init(NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	double *x = NULL;
	int xn = DEFAULT_VECTOR_X_LEN;
	double alpha;

	starpu_data_handle_t x_handle;

	if (argc > 1)
	{
		xn = atoi(argv[1]);
		if (xn < 1)
		{
			fprintf(stderr, "invalid vector length %d\n", xn);
			exit(EXIT_FAILURE);
		}
	}

	printf("Vector 'x' len: %d\n", xn);

	x = malloc(xn * sizeof(x[0]));
	if (x == NULL)
	{
		fprintf(stderr, "memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	vector_init(x, xn);
	printf("Initial vector 'x':\n");
	vector_disp(x, xn);
	printf("\n");

	alpha = ((double)(rand() % 100) / 10);
	printf("Factor 'alpha': %.3lf\n", alpha);
	printf("\n");

	// vector_scale(x, xn, alpha); // fonction cpu à imiter avec STARPU

	starpu_vector_data_register(&x_handle, STARPU_MAIN_RAM, (uintptr_t)x, xn, sizeof(x[0]));

	starpu_task_insert(
		&mult_codelet,
		STARPU_RW, x_handle,
		STARPU_VALUE, &alpha, sizeof(alpha),
		0);

	starpu_task_wait_for_all();
	starpu_data_unregister(x_handle);

	// scal_cpu_func

	printf("Scaled vector 'x':\n");
	vector_disp(x, xn);

	printf("\n");

	free(x);

	starpu_shutdown();

	return 0;
}
