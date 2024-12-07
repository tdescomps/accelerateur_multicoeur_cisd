#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define N 1000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		out[idx] = a[idx] + b[idx];
	}
}

int main()
{
	float *a, *b, *out;
	float *d_a, *d_b, *d_out;

	// Allocate memory
	a = (float *)malloc(sizeof(float) * N);
	b = (float *)malloc(sizeof(float) * N);
	out = (float *)malloc(sizeof(float) * N);

	// Allocate memory on GPU
	cudaMalloc((void **)&d_a, sizeof(float) * N);
	cudaMalloc((void **)&d_b, sizeof(float) * N);
	cudaMalloc((void **)&d_out, sizeof(float) * N);

	// Initialize array
	for (int i = 0; i < N; i++)
	{
		a[i] = i * 1.0f;
		b[i] = i * 2.0f;
	}

	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

	// Main function
	int block_size = 256;
    int grid_size = ((N + block_size - 1) / block_size);
    vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);

	cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

	// Verification
	for (int i = 0; i < N; i++)
	{
		if (fabs(out[i] - a[i] - b[i]) > MAX_ERR)
		{
			printf("error\n");
			break;
		}
	}

	printf("out[2] = %f\n", out[2]);

	free(a);
	free(b);
	free(out);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);

	return 0;
}
