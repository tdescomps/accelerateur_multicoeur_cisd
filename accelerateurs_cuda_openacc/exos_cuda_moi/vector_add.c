#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define N 1000000
#define MAX_ERR 1e-6

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    // Main function
    vector_add(out, a, b, N);

    // Verification
    for(int i = 0; i < N; i++){
        if (fabs(out[i] - a[i] - b[i]) > MAX_ERR) {
			printf("error\n");
			break;
		}
    }

    printf("out[2] = %f\n", out[2]);

	free(a);
	free(b);
	free(out);

	return 0;
}
