#include <stdio.h>

#define N 32

__global__ void add(int *a, int *b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void fill_int(int *array_to_fill, int size)
{
    for (int i = 0; i < size; ++i)
    {
        array_to_fill[i] = i;
    }
}

void print_array(int* array, int size) {
    printf("[ ");
    for (int i = 0; i < size; ++i)
    {
        printf("%d ", array[i]);
    }
    printf("]\n");
}

int main(void)
{
    int *a, *b, *c;
    int *gpu_a, *gpu_b, *gpu_c;
    int size = N * sizeof(int);
    // allocation de l’espace pour le device
    cudaMalloc((void **)&gpu_a, size);
    cudaMalloc((void **)&gpu_b, size);
    cudaMalloc((void **)&gpu_c, size);
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    fill_int(a, N);
    fill_int(b, N);

    // Copie des donnees vers le Device
    cudaMemcpy(gpu_a, &a, size, cudaMemcpyHostToDevice);
    // checkCudaErrors(cudaMemcpy(gpu_a, &a, size, cudaMemcpyHostToDevice));
    cudaMemcpy(gpu_b, &b, size, cudaMemcpyHostToDevice);
    add<<<N, 1>>>(gpu_a, gpu_b, gpu_c);
    // Copie du resultat vers Host
    cudaMemcpy(&c, gpu_c, size, cudaMemcpyDeviceToHost);

    // print_array(c, N);

    // Liberation de l’espace alloue
    free(a);
    free(b);
    free(c);
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    return 0;
}