#include <stdio.h>

__global__ void cuda_hello() {
	printf("hello\n");
}

int main(int argc, char* argv[]) {
	cuda_hello<<<1,1>>>();
	return 0;
}