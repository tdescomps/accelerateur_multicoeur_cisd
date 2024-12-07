#include <stdlib.h>
#include <stdio.h>
#include <starpu.h>

#define DEFAULT_VECTOR_X_LEN 100
#define DEFAULT_NB_PARTS 1

void vector_init(double *vector, int len)
{
	int i;
	for (i=0; i<len; i++)
	{
		vector[i] = (double)(rand() % 100);
	}
}

void vector_disp(double *vector, int len)
{
	int i;
	for (i=0; i<len; i++)
	{
		printf("[%d]: %.3lf\n", i, vector[i]);
	}
}

void vector_scale(double *vector, int len, double factor)
{
	int i;
	fprintf(stderr, "vector_scale task: working on %d element(s)\n", len);
	for (i=0; i<len; i++)
	{
		vector[i] *= factor;	
	}
}

void vector_scale_func(void *buffers[], void *cl_args)
{
	struct starpu_vector_interface *vector_handle = buffers[0];

	double *vector = (double *)STARPU_VECTOR_GET_PTR(vector_handle);
	int len = (int)STARPU_VECTOR_GET_NX(vector_handle);

	double factor;
	starpu_codelet_unpack_args(cl_args, &factor);

	vector_scale(vector, len, factor);
}

struct starpu_codelet vector_scale_cl =
{
	.cpu_funcs = { vector_scale_func },
	.nbuffers = 1,
	.modes = { STARPU_RW },
};

int main(int argc, char *argv[])
{
	double *x = NULL;
	int xn = DEFAULT_VECTOR_X_LEN;
	int nb_parts = DEFAULT_NB_PARTS;
	double alpha;
	int ret;

	ret = starpu_init(NULL);
	if (ret != 0)
	{
		exit(EXIT_FAILURE);
	}

	if (argc > 1)
	{
		xn = atoi(argv[1]);
		if (xn < 1)
		{
			fprintf(stderr, "invalid vector length %d\n", xn);
			starpu_shutdown();
			exit(EXIT_FAILURE);
		}
	}
	if (argc > 2)
	{
		nb_parts = atoi(argv[2]);
		if (nb_parts < 1 || nb_parts >= xn)
		{
			fprintf(stderr, "invalid number of parts %d\n", nb_parts);
			starpu_shutdown();
			exit(EXIT_FAILURE);
		}
	}
	printf("Vector 'x' len: %d\n", xn);

	x = malloc(xn * sizeof(x[0]));
	if (x == NULL)
	{
		fprintf(stderr, "memory allocation failed\n");
		starpu_shutdown();
		exit(EXIT_FAILURE);
	}

	vector_init(x, xn);
	printf("Initial vector 'x':\n");
	vector_disp(x, xn);
	printf("\n");

	alpha = ((double)(rand() % 100)/10);
	printf("Factor 'alpha': %.3lf\n", alpha);
	printf("\n");

	{
		starpu_data_handle_t x_handle;
		starpu_vector_data_register(&x_handle, STARPU_MAIN_RAM, (uintptr_t)x, xn, sizeof(x[0]));

		struct starpu_data_filter f_part =
		{
			.filter_func = starpu_vector_filter_block,
			.nchildren = nb_parts,
		};

		starpu_data_handle_t sub_x_handles[nb_parts];
		starpu_data_partition_plan(x_handle, &f_part, sub_x_handles);

		starpu_data_partition_submit(x_handle, nb_parts, sub_x_handles);
		int i;
		for (i=0; i<nb_parts; i++)
		{
			starpu_task_insert(&vector_scale_cl, STARPU_RW, sub_x_handles[i], STARPU_VALUE, &alpha, sizeof(alpha), 0);
		}
		starpu_data_unpartition_submit(x_handle, nb_parts, sub_x_handles, -1);

		starpu_task_wait_for_all();

		starpu_data_partition_clean(x_handle, nb_parts, sub_x_handles);

		starpu_data_unregister(x_handle);
	}

	printf("Scaled vector 'x':\n");
	vector_disp(x, xn);
	printf("\n");

	free(x);

	starpu_shutdown();

	return 0;
}
