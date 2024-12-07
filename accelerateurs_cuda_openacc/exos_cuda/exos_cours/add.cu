#include <stdio.h>

__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int main(void)
{
    int a, b, c;
    int *gpu_a, *gpu_b, *gpu_c;
    int size = sizeof(int);
    // allocation de l’espace pour le device
    cudaMalloc((void **)&gpu_a, size);
    cudaMalloc((void **)&gpu_b, size);
    cudaMalloc((void **)&gpu_c, size);
    a = 2;
    b = 7;

    // Copie des donnees vers le Device
    cudaMemcpy(gpu_a, &a, size, cudaMemcpyHostToDevice);
    // checkCudaErrors(cudaMemcpy(gpu_a, &a, size, cudaMemcpyHostToDevice));
    cudaMemcpy(gpu_b, &b, size, cudaMemcpyHostToDevice);
    add<<<1, 1>>>(gpu_a, gpu_b, gpu_c);
    // Copie du resultat vers Host
    cudaMemcpy(&c, gpu_c, size, cudaMemcpyDeviceToHost);
    // Liberation de l’espace alloue
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    printf("Computed on device: %d + %d = %d\n", a, b, c);

    return 0;
}