#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define MAX_DISP_ARRAY 10

void usage (void)
{
	fprintf(stderr, "demo [TESTNAME]\n");
	fprintf(stderr, "TESTNAME values:\n");
	fprintf(stderr, "- info\n");
	fprintf(stderr, "- set0\n");
	fprintf(stderr, "- set1\n");
	fprintf(stderr, "- set\n");
	fprintf(stderr, "- add\n");
#if defined(__INTEL_COMPILER)
	fprintf(stderr, "- exp (Intel compiler)\n");
#else
	fprintf(stderr, "! unavailable: exp (Intel compiler only)\n");
#endif
	fprintf(stderr, "- fmadd\n");
	fprintf(stderr, "- cmp\n");

	exit(EXIT_FAILURE);
}

#define REG_BYTES (sizeof(__m256))
#define REG_ELEMENT_TYPE float
#define REG_ELEMENT_TYPE_STR "float"
#define REG_NB_ELEMENTS (REG_BYTES / sizeof(REG_ELEMENT_TYPE))

void print_format(void)
{
	printf("|");
	int i;
	for (i=0; i<REG_NB_ELEMENTS; i++)
	{
		if (i > 0)
		{
			if (i % (REG_NB_ELEMENTS / 2) == 0)
			{
				printf("   ::  ");
			}
			else if (i % (REG_NB_ELEMENTS / 4) == 0)
			{
				printf("  : ");
			}
			else
			{
				printf(" .");
			}
		}
		printf(" %lu", REG_NB_ELEMENTS-i-1);
	}

	printf(" |");
}
void print_reg(__m256 r)
{
	printf("|");
	int i;
	for (i=0; i<REG_NB_ELEMENTS; i++)
	{
		if (i > 0)
		{
			if (i % (REG_NB_ELEMENTS / 2) == 0)
			{
				printf("   ::  ");
			}
			else if (i % (REG_NB_ELEMENTS / 4) == 0)
			{
				printf("  : ");
			}
			else
			{
				printf(" .");
			}
		}
		printf(" %4.1f", r[REG_NB_ELEMENTS-i-1]);
	}

	printf(" |");
}

void print_reg_hex(__m256 r)
{
	uint32_t ri[REG_NB_ELEMENTS];
	_mm256_storeu_ps((REG_ELEMENT_TYPE *)ri, r);
	printf("|");
	int i;
	for (i=0; i<REG_NB_ELEMENTS; i++)
	{
		if (i > 0)
		{
			if (i % (REG_NB_ELEMENTS / 2) == 0)
			{
				printf("   ::  ");
			}
			else if (i % (REG_NB_ELEMENTS / 4) == 0)
			{
				printf("  : ");
			}
			else
			{
				printf(" .");
			}
		}
		printf(" 0x%08x", ri[REG_NB_ELEMENTS-i-1]);
	}

	printf(" |");
}

void print_array(float v[8])
{
	int i;
	for (i=0; i<8; i++)
	{
		if (i>0)
		{
			printf(",");
		}
		printf(" %4.1f", v[i]);
	}
}

void print_array_reverse(float v[8])
{
	int i;
	for (i=0; i<8; i++)
	{
		if (i>0)
		{
			printf(",");
		}
		printf(" %4.1f", v[7-i]);
	}
}

__attribute__((noinline)) void set0(__m256 *x)
{
	*x = _mm256_setzero_ps();
}

__attribute__((noinline)) void set1(__m256 *x, float v)
{
	*x = _mm256_set1_ps(v);
}

__attribute__((noinline)) void set(__m256 *x, float v[8])
{
	*x = _mm256_set_ps(v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0]);
}

__attribute__((noinline)) void add(__m256 x, __m256 y, __m256 *z)
{
	*z = _mm256_add_ps(x, y);
}

#if defined(__INTEL_COMPILER)
__attribute__((noinline)) void exp_func(__m256 x, __m256 *z)
{
	*z = _mm256_exp_ps(x);
}
#endif

__attribute__((noinline)) void cmpgt(__m256 x, __m256 y, __m256 *z)
{
	*z = _mm256_cmp_ps(x, y, _CMP_GT_OS);
}
__attribute__((noinline)) void fmadd(__m256 w, __m256 x, __m256 y, __m256 *z)
{
	*z = _mm256_fmadd_ps(w, x, y);
}


int main(int argc, char *argv[])
{
	{
		int i = 1;
		while (i < argc)
		{
			char *arg = argv[i];
			if (strcmp(arg, "info") == 0)
			{
				printf("Infos:\n\n");
				printf("- AVX2\n");
				printf("\n");
				printf("- registers\n");
				printf("  - %lu bits\n", 8*sizeof(__m256));
				printf("  - %lu bytes\n", sizeof(__m256));
				printf("  - '%s' element type\n", REG_ELEMENT_TYPE_STR);
				printf("  - %lu elements per register\n", REG_NB_ELEMENTS);
				printf("\n");
				printf("- register display format\n");
				print_format();
				printf("\n\n");
			}
			else if (strcmp(arg, "set0") == 0)
			{
				/* set0 */
				printf("_mm256_setzero_ps:\n\n");

				__m256 x;
				set0(&x);

				printf("-->      ");
				print_reg(x);
				printf("\n\n");
			}
			else if (strcmp(arg, "set1") == 0)
			{
				/* set1 */
				float v = 3.1;
				printf("_mm256_set1_ps(%3.1f):\n\n", v);

				__m256 x;
				set1(&x, v);

				printf("-->      ");
				print_reg(x);
				printf("\n\n");
			}
			else if (strcmp(arg, "set") == 0)
			{
				/* set */
				float v[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
				printf("_mm256_set_ps(");
				print_array_reverse(v);
				printf(" ):\n\n");

				__m256 x;
				set(&x, v);

				printf("-->      ");
				print_reg(x);
				printf("\n\n");
			}
			else if (strcmp(arg, "add") == 0)
			{
				float vx[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
				float vy[8] = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

				__m256 x;
				__m256 y;
				__m256 z;

				set(&x, vx);
				set(&y, vy);

				printf("_mm256_add_ps(");
				print_reg(x);
				printf(",\n");
				printf("              ");
				print_reg(y);
				printf("):\n\n");

				add(x, y, &z);
				printf("-->           ");
				print_reg(z);
				printf("\n\n");
			}
#if defined(__INTEL_COMPILER)
			else if (strcmp(arg, "exp") == 0)
			{
				float vx[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.1};

				__m256 x;
				__m256 z;

				set(&x, vx);

				printf("_mm256_exp_ps(");
				print_reg(x);
				printf("):\n\n");

				exp_func(x, &z);
				printf("-->           ");
				print_reg(z);
				printf("\n\n");
			}
#endif
			else if (strcmp(arg, "fmadd") == 0)
			{
				float vw[8] = { 3.6, -8.9,  8.1,  2.0,  0.8, -6.9,  4.5, -1.7 };
				float vx[8] = { 1.6,  0.5,  2.1,  5.8, -3.6,  8.6,  9.6, -6.7 };
				float vy[8] = {-1.5, -6.9,  1.3, -4.7,  7.3,  0.1,  9.6, -6.5 };

				__m256 w;
				__m256 x;
				__m256 y;
				__m256 z;

				set(&w, vw);
				set(&x, vx);
				set(&y, vy);

				printf("_mm256_fmadd_ps(");
				print_reg(w);
				printf(",\n");
				printf("                ");
				print_reg(x);
				printf(",\n");
				printf("                ");
				print_reg(y);
				printf("):\n\n");

				fmadd(w, x, y, &z);
				printf("-->             ");
				print_reg(z);
				printf("\n\n");
			}
			else if (strcmp(arg, "cmp") == 0)
			{
				float vx[8] = {2.2, 0.5, 9.4, 2.7, 9.0, 2.4, 5.4, 0.4};
				float vy[8] = {9.7, 2.0, 0.3, 1.8, 0.0, 4.2, 2.2, 6.0};

				__m256 x;
				__m256 y;
				__m256 z;

				set(&x, vx);
				set(&y, vy);

				printf("_mm256_cmp_ps(");
				print_reg(x);
				printf(",\n");
				printf("              ");
				print_reg(y);
				printf(",\n");
				printf("              _CMP_GT_OS /* '>' */):\n\n");

				cmpgt(x, y, &z);
				printf("-->           ");
				print_reg_hex(z);
				printf("\n\n");
			}
			else if (strcmp(arg, "minmax") == 0)
			{
				/* minmax */
				float x[8] = {0, 1, 2, 3, 4, 5, 6, 7};
				float y[8] = {1, 2, 1, 2, 4, 1, 8, 1};

				__m256 reg_x; //= _mm256_load_ps(x);
				__m256 reg_y; //= _mm256_load_ps(y);

				set(&reg_x, x);
				set(&reg_y, y);

				
				printf("fonction _mm256_max_ps()\n");

				__m256 result = _mm256_max_ps(reg_x, reg_y);

				print_reg(result);
				printf("\n");

				printf("fonction _mm256_min_ps()\n");

				result = _mm256_min_ps(reg_x, reg_y);

				print_reg(result);
				printf("\n\n");


				__m256 gt = _mm256_cmp_ps(reg_x, reg_y, 13);
				//__m256 gt = _mm256_cmp_ps

				printf("mon max\n");

				result = _mm256_fmadd_ps(gt, reg_x, reg_y);

				print_reg(gt);
				printf("\n");
				print_reg(result);
				printf("\n");

				printf("mon min\n");

				result = _mm256_min_ps(reg_x, reg_y);

				print_reg(result);


				printf("\n\n");
			}
			else
			{
				usage();
			}
			i++;
		}

		if (i == 1)
		{
			usage();
		}
	}

	return 0;
}

