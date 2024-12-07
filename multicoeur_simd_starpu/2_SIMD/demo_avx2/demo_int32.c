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
	fprintf(stderr, "- cmpgt\n");
	fprintf(stderr, "- xor\n");
	fprintf(stderr, "- slli\n");
	fprintf(stderr, "- shuffle\n");
	fprintf(stderr, "- permutevar\n");
	fprintf(stderr, "- blend\n");
	fprintf(stderr, "- extract\n");
	fprintf(stderr, "- extract128\n");
	fprintf(stderr, "- insert\n");
	fprintf(stderr, "- packs\n");
	fprintf(stderr, "- unpacklo\n");
	fprintf(stderr, "- unpackhi\n");
	fprintf(stderr, "- gather\n");
	fprintf(stderr, "- broadcast\n");

	exit(EXIT_FAILURE);
}

#define REG_BYTES (sizeof(__m256i))
#define REG_ELEMENT_TYPE int32_t
#define REG_ELEMENT_TYPE_STR "int32_t"
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
void print_reg(__m256i r)
{
	uint32_t ri[REG_NB_ELEMENTS];
	_mm256_storeu_si256((__m256i *)ri, r);
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
		printf(" %4d", ri[REG_NB_ELEMENTS-i-1]);
	}

	printf(" |");
}

void print_half_reg(__m128i r)
{
	uint32_t ri[REG_NB_ELEMENTS];
	_mm_store_si128((__m128i *)ri, r);
	printf("|");
	int i;
	for (i=0; i<REG_NB_ELEMENTS/2; i++)
	{
		if (i > 0)
		{
			if (i % (REG_NB_ELEMENTS / 4) == 0)
			{
				printf("  : ");
			}
			else
			{
				printf(" .");
			}
		}
		printf(" %4d", ri[REG_NB_ELEMENTS/2-i-1]);
	}

	printf(" |");
}

void print_half_reg_hex(__m128i r)
{
	uint32_t ri[REG_NB_ELEMENTS];
	_mm_store_si128((__m128i *)ri, r);
	printf("|");
	int i;
	for (i=0; i<REG_NB_ELEMENTS/2; i++)
	{
		if (i > 0)
		{
			if (i % (REG_NB_ELEMENTS / 4) == 0)
			{
				printf("  : ");
			}
			else
			{
				printf(" .");
			}
		}
		printf(" 0x%08x", ri[REG_NB_ELEMENTS/2-i-1]);
	}

	printf(" |");
}

void print_reg_hex(__m256i r)
{
	uint32_t ri[REG_NB_ELEMENTS];
	_mm256_storeu_si256((__m256i *)ri, r);
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

void print_array(int32_t v[8])
{
	int i;
	for (i=0; i<8; i++)
	{
		if (i>0)
		{
			printf(",");
		}
		printf(" %4d", v[i]);
	}
}

void print_array_reverse(int32_t v[8])
{
	int i;
	for (i=0; i<8; i++)
	{
		if (i>0)
		{
			printf(",");
		}
		printf(" %4d", v[7-i]);
	}
}

void print_data(int32_t *data, int n)
{
	int i;
	for (i=0; i<n; i++)
	{
		if (i>0)
		{
			printf(",");
		}
		printf(" %2d", data[i]);
	}
}


__attribute__((noinline)) void set0(__m256i *x)
{
	*x = _mm256_setzero_si256();
}

__attribute__((noinline)) void set1(__m256i *x, int32_t v)
{
	*x = _mm256_set1_epi32(v);
}

__attribute__((noinline)) void set(__m256i *x, int32_t v[8])
{
	*x = _mm256_set_epi32(v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0]);
}

__attribute__((noinline)) void set_half_reg(__m128i *x, int32_t v[4])
{
	*x = _mm_set_epi32(v[3], v[2], v[1], v[0]);
}

__attribute__((noinline)) void add(__m256i x, __m256i y, __m256i *z)
{
	*z = _mm256_add_epi32(x, y);
}

__attribute__((noinline)) void cmpgt(__m256i x, __m256i y, __m256i *z)
{
	*z = _mm256_cmpgt_epi32(x, y);
}

__attribute__((noinline)) void xor(__m256i x, __m256i y, __m256i *z)
{
	*z = _mm256_xor_si256(x, y);
}

__attribute__((noinline)) void slli(__m256i x, __m256i *y, int v)
{
	*y = _mm256_slli_epi32(x, v);
}

__attribute__((noinline)) void shuffle_1233(__m256i x, __m256i *y)
{
	/* Note: shuffle 32-bit integers in 128-bit lanes */
	/* Note: control must be a constant known at compile-time */
	*y = _mm256_shuffle_epi32(x, _MM_SHUFFLE(1,2,3,3));
}

__attribute__((noinline)) void shuffle_0231(__m256i x, __m256i *y)
{
	/* Note: shuffle 32-bit integers in 128-bit lanes */
	/* Note: control must be a constant known at compile-time */
	*y = _mm256_shuffle_epi32(x, _MM_SHUFFLE(0,2,3,1));
}

__attribute__((noinline)) void permutevar(__m256i x, __m256i *y, __m256i v)
{
	*y = _mm256_permutevar8x32_epi32(x, v);
}

__attribute__((noinline)) void blend_0x53(__m256i x, __m256i y, __m256i *z)
{
	/* Note: control must be a constant known at compile-time */
	*z = _mm256_blend_epi32(x, y, 0x53);
}

__attribute__((noinline)) void extract_0(__m256i x, int32_t *y)
{
	/* Note: index must be a constant known at compile-time */
	*y = _mm256_extract_epi32(x, 0);
}

__attribute__((noinline)) void extract_3(__m256i x, int32_t *y)
{
	/* Note: index must be a constant known at compile-time */
	*y = _mm256_extract_epi32(x, 3);
}

__attribute__((noinline)) void extract_i128_0(__m256i x, __m128i *y)
{
	/* Note: index must be a constant known at compile-time */
	*y = _mm256_extracti128_si256(x, 0);
}

__attribute__((noinline)) void extract_i128_1(__m256i x, __m128i *y)
{
	/* Note: index must be a constant known at compile-time */
	*y = _mm256_extracti128_si256(x, 1);
}

__attribute__((noinline)) void insert_2(__m256i x, int32_t y, __m256i *z)
{
	/* Note: index must be a constant known at compile-time */
	*z = _mm256_insert_epi32(x, y, 2);
}

__attribute__((noinline)) void insert_5(__m256i x, int32_t y, __m256i *z)
{
	/* Note: index must be a constant known at compile-time */
	*z = _mm256_insert_epi32(x, y, 5);
}

__attribute__((noinline)) void packs(__m256i x, __m256i y, __m256i *z)
{
	*z = _mm256_packs_epi32(x, y);
}

__attribute__((noinline)) void unpacklo(__m256i x, __m256i y, __m256i *z)
{
	*z = _mm256_unpacklo_epi32(x, y);
}

__attribute__((noinline)) void unpackhi(__m256i x, __m256i y, __m256i *z)
{
	*z = _mm256_unpackhi_epi32(x, y);
}

__attribute__((noinline)) void gather4(const int32_t *data, __m256i i, __m256i *z)
{
	*z = _mm256_i32gather_epi32(data, i, 4);
}

__attribute__((noinline)) void gather8(const int32_t *data, __m256i i, __m256i *z)
{
	*z = _mm256_i32gather_epi32(data, i, 8);
}

__attribute__((noinline)) void broadcast(__m128i x, __m256i *y)
{
	*y = _mm256_broadcastd_epi32(x);
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
				printf("  - %lu bits\n", 8*sizeof(__m256i));
				printf("  - %lu bytes\n", sizeof(__m256i));
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
				printf("_mm256_setzero_si256:\n\n");

				__m256i x;
				set0(&x);

				printf("-->      ");
				print_reg(x);
				printf("\n\n");
			}
			else if (strcmp(arg, "set1") == 0)
			{
				/* set1 */
				int32_t v = 3;
				printf("_mm256_set1_epi32(%4d):\n\n", v);

				__m256i x;
				set1(&x, v);

				printf("-->      ");
				print_reg(x);
				printf("\n\n");
			}
			else if (strcmp(arg, "set") == 0)
			{
				/* set */
				int32_t v[8] = {0, 1, 2, 3, 4, 5, 6, 7};
				printf("_mm256_set_epi32(");
				print_array_reverse(v);
				printf(" ):\n\n");

				__m256i x;
				set(&x, v);

				printf("-->      ");
				print_reg(x);
				printf("\n\n");
			}
			else if (strcmp(arg, "add") == 0)
			{
				int32_t vx[8] = {  2,  15,  -7,   7, -12,   4,  -1,  -3};
				int32_t vy[8] = { -7,  18,   3,   4,  -1, -19,  11, -10};

				__m256i x;
				__m256i y;
				__m256i z;

				set(&x, vx);
				set(&y, vy);

				printf("_mm256_add_epi32(");
				print_reg(x);
				printf(",\n");
				printf("                 ");
				print_reg(y);
				printf("):\n\n");

				add(x, y, &z);
				printf("-->              ");
				print_reg(z);
				printf("\n\n");
			}
			else if (strcmp(arg, "cmpgt") == 0)
			{
				int32_t vx[8] = {2, 0, 9, 2, 9, 2, 5, 0};
				int32_t vy[8] = {9, 2, 0, 1, 0, 4, 2, 6};

				__m256i x;
				__m256i y;
				__m256i z;

				set(&x, vx);
				set(&y, vy);

				printf("_mm256_cmpgt_epi32(");
				print_reg(x);
				printf(",\n");
				printf("                   ");
				print_reg(y);
				printf(") /* '>' */:\n\n");

				cmpgt(x, y, &z);
				printf("-->                ");
				print_reg_hex(z);
				printf("\n\n");
			}
			else if (strcmp(arg, "xor") == 0)
			{
				int32_t vx[8] = {878763931, -2060451143, 1986155509, 1494103813, 1319016401,
					605332096,  -289232084,  250863117};
				int32_t vy[8] = {-1865704863, 1945693675, -1050219988,  -943158044, 1533449600,
				       1334506605,  -521535371,  256014394};

				__m256i x;
				__m256i y;
				__m256i z;

				set(&x, vx);
				set(&y, vy);

				printf("_mm256_xor_si256(");
				print_reg_hex(x);
				printf(",\n");
				printf("                 ");
				print_reg_hex(y);
				printf("):\n\n");

				xor(x, y, &z);
				printf("-->              ");
				print_reg_hex(z);
				printf("\n\n");
			}
			else if (strcmp(arg, "slli") == 0)
			{
				int32_t vx[8] = {1253760006,  585647103, 1988866074,  595929849,  282787286,
					1029668122,  912124790, 1827584937};

				__m256i x;
				__m256i y;
				int v = 12;

				set(&x, vx);

				printf("_mm256_slli_epi32(");
				print_reg_hex(x);
				printf(", %d):\n\n", v);

				slli(x, &y, v);
				printf("-->               ");
				print_reg_hex(y);
				printf("\n\n");
			}
			else if (strcmp(arg, "shuffle") == 0)
			{
				int32_t vx[8] = {30, 41, 58, 57, 60, 25, 39, 43};

				__m256i x;
				__m256i y;

				set(&x, vx);

				{
					printf("_mm256_shuffle_epi32(");
					print_reg(x);
					printf(",\n");
					printf("                     _MM_SHUFFLE(1,2,3,3)):\n\n");

					shuffle_1233(x, &y);
					printf("-->                  ");
					print_reg(y);
					printf("\n\n");
				}
				{
					printf("_mm256_shuffle_epi32(");
					print_reg(x);
					printf(",\n");
					printf("                     _MM_SHUFFLE(0,2,3,1)):\n\n");

					shuffle_0231(x, &y);
					printf("-->                  ");
					print_reg(y);
					printf("\n\n");
				}
			}
			else if (strcmp(arg, "permutevar") == 0)
			{
				int32_t vx[8] = {31, 54, 22, 66, 30, 14, 24, 17};
				int32_t vv[8] = {1, 7, 0, 1, 2, 6, 5, 0};

				__m256i x;
				__m256i y;
				__m256i v;

				set(&x, vx);
				set(&v, vv);

				printf("_mm256_permutevar8x32_epi32(");
				print_reg(x);
				printf(",\n");
				printf("                           ");
				print_reg(v);
				printf("):\n\n");

				permutevar(x, &y, v);
				printf("-->                        ");
				print_reg(y);
				printf("\n\n");
			}
			else if (strcmp(arg, "blend") == 0)
			{
				int32_t vx[8] = {13, 23, 17, 35, 34, 14, 16, 36};
				int32_t vy[8] = {49, 14, 34, 55, 58, 19, 65, 16};

				__m256i x;
				__m256i y;
				__m256i z;

				set(&x, vx);
				set(&y, vy);

				printf("_mm256_blend_epi32(");
				print_reg(x);
				printf(",\n");
				printf("                   ");
				print_reg(y);
				printf(",\n");
				printf("                   0x%02x):\n\n", 0x53);

				blend_0x53(x, y, &z);
				printf("-->                ");
				print_reg(z);
				printf("\n\n");
			}
			else if (strcmp(arg, "extract") == 0)
			{
				int32_t vx[8] = {42, 18, 68, 19, 19, 65, 49, 39};

				__m256i x;

				set(&x, vx);

				{
					printf("_mm256_extract_epi32(");
					print_reg(x);
					printf(",\n");
					printf("                     %d):\n\n", 3);

					int32_t y;
					extract_3(x, &y);
					printf("-->                  %4d", y);
					printf("\n\n");
				}
				{
					printf("_mm256_extract_epi32(");
					print_reg(x);
					printf(",\n");
					printf("                     %d):\n\n", 0);

					int32_t y;
					extract_0(x, &y);
					printf("-->                 %4d", y);
					printf("\n\n");
				}
			}
			else if (strcmp(arg, "extract128") == 0)
			{
				int32_t vx[8] = {69, 22, 66, 65, 51, 47, 34, 59};

				__m256i x;
				__m128i y;

				set(&x, vx);

				{
					printf("_mm256_extracti128_si256(");
					print_reg(x);
					printf(",\n");
					printf("                         %d):\n\n", 1);

					extract_i128_1(x, &y);
					printf("-->                      ");
					print_half_reg(y);
					printf("\n\n");
				}
				{
					printf("_mm256_extracti128_si256(");
					print_reg(x);
					printf(",\n");
					printf("                         %d):\n\n", 0);

					extract_i128_0(x, &y);
					printf("-->                      ");
					print_half_reg(y);
					printf("\n\n");
				}
			}
			else if (strcmp(arg, "insert") == 0)
			{
				int32_t vx[8] = {53, 57, 13, 28, 55, 32, 25, 65};

				__m256i x;

				set(&x, vx);

				{
					int32_t y = 42;
					printf("_mm256_insert_epi32(");
					print_reg(x);
					printf(",\n");
					printf("                    %d\n", y);
					printf("                    %d):\n\n", 2);

					__m256i z;
					insert_2(x, y, &z);
					printf("-->                 ");
					print_reg(z);
					printf("\n\n");
				}
				{
					int32_t y = 42;
					printf("_mm256_insert_epi32(");
					print_reg(x);
					printf(",\n");
					printf("                    %d\n", y);
					printf("                    %d):\n\n", 5);

					__m256i z;
					insert_5(x, y, &z);
					printf("-->                 ");
					print_reg(z);
					printf("\n\n");
				}
			}
			else if (strcmp(arg, "packs") == 0)
			{
				int32_t vx[8] = {62, 37, 63, 39, 22, 36, 29, 68};
				int32_t vy[8] = {59, 58, 67, 21, 11, 65, 50, 69};

				__m256i x;
				__m256i y;
				__m256i z;

				set(&x, vx);
				set(&y, vy);

				printf("_mm256_packs_epi32(");
				print_reg_hex(x);
				printf(",\n");
				printf("                   ");
				print_reg_hex(y);
				printf("):\n\n");

				packs(x, y, &z);
				printf("-->                ");
				print_reg_hex(z);
				printf("\n\n");
			}
			else if (strcmp(arg, "unpacklo") == 0)
			{
				int32_t vx[8] = {69, 46, 30, 56, 51, 17, 30, 29};
				int32_t vy[8] = {48, 32, 46, 62, 50, 38, 67, 51};

				__m256i x;
				__m256i y;
				__m256i z;

				set(&x, vx);
				set(&y, vy);

				printf("_mm256_unpacklo_epi32(");
				print_reg_hex(x);
				printf(",\n");
				printf("                      ");
				print_reg_hex(y);
				printf("):\n\n");

				unpacklo(x, y, &z);
				printf("-->                   ");
				print_reg_hex(z);
				printf("\n\n");
			}
			else if (strcmp(arg, "unpackhi") == 0)
			{
				int32_t vx[8] = {10, 38, 61, 38, 52, 43, 53, 59};
				int32_t vy[8] = {51, 69, 20, 15, 60, 12, 39, 44};

				__m256i x;
				__m256i y;
				__m256i z;

				set(&x, vx);
				set(&y, vy);

				printf("_mm256_unpackhi_epi32(");
				print_reg_hex(x);
				printf(",\n");
				printf("                      ");
				print_reg_hex(y);
				printf("):\n\n");

				unpackhi(x, y, &z);
				printf("-->                   ");
				print_reg_hex(z);
				printf("\n\n");
			}
			else if (strcmp(arg, "gather") == 0)
			{
				int32_t data[24] = {38, 52, 26, 33, 26, 67, 50, 50, 35, 66, 40, 27, 58, 50, 27, 53, 15, 18, 33, 34, 16, 15, 68, 32};

				__m256i i;
				__m256i z;

				{
					int32_t vi[8] = { 0,  1,  2, 3, 5,  6,  7,  8};
					set(&i, vi);

					printf("_mm256_i32gather_epi32([");
					print_data(data, 24);
					printf("],\n");
					printf("                       ");
					print_reg(i);
					printf(",\n");
					printf("                       %d", 4);
					printf("):\n\n");

					gather4(data, i, &z);
					printf("-->                    ");
					print_reg(z);
					printf("\n\n");
				}
				{
					int32_t vi[8] = { 0,  1,  2, 3, 5,  6,  7,  8};
					set(&i, vi);
					printf("_mm256_i32gather_epi32([");
					print_data(data, 24);
					printf("],\n");
					printf("                       ");
					print_reg(i);
					printf(",\n");
					printf("                       %d", 8);
					printf("):\n\n");

					gather8(data, i, &z);
					printf("-->                    ");
					print_reg(z);
					printf("\n\n");
				}

				{
					int32_t vi[8] = { 9,  2,  1, 3, 8,  11,  7,  6};
					set(&i, vi);

					printf("_mm256_i32gather_epi32([");
					print_data(data, 24);
					printf("],\n");
					printf("                       ");
					print_reg(i);
					printf(",\n");
					printf("                       %d", 4);
					printf("):\n\n");

					gather4(data, i, &z);
					printf("-->                    ");
					print_reg(z);
					printf("\n\n");
				}

				{
					int32_t vi[8] = { 9,  2,  1, 3, 8,  11,  7,  6};
					set(&i, vi);

					printf("_mm256_i32gather_epi32([");
					print_data(data, 24);
					printf("],\n");
					printf("                       ");
					print_reg(i);
					printf(",\n");
					printf("                       %d", 8);
					printf("):\n\n");

					gather8(data, i, &z);
					printf("-->                    ");
					print_reg(z);
					printf("\n\n");
				}
			}
			else if (strcmp(arg, "broadcast") == 0)
			{
				int32_t vx[8] = {42, 51, 50, 23};

				__m128i x;
				__m256i y;

				set_half_reg(&x, vx);

				printf("_mm256_broadcastd_epi32(");
				print_half_reg(x);
				printf("):\n\n");

				broadcast(x, &y);
				printf("-->                     ");
				print_reg(y);
				printf("\n\n");
			}
			/* else if (strcmp(arg, "minmax") == 0)
			{
				int32_t x[8] = {0, 1, 2, 3, 4, 5, 6, 7};
				int32_t y[8] = {1, 2, 1, 2, 4, 1, 8, 1};

				__m256i reg_x; //= _mm256_load_epi32(x);
				__m256i reg_y; //= _mm256_load_epi32(y);


				set(&reg_x, x);
				set(&reg_y, y);

				
				printf("fonction _mm256_max_epi32()\n");

				__m256i result = _mm256_max_epi32(reg_x, reg_y);

				print_reg(result);

				printf("fonction _mm256_min_epi32()\n");

				result = _mm256_min_epi32(reg_x, reg_y);

				print_reg(result);


				__m256i gt = _mm256_cmpgt_epi32(reg_x, reg_y);

				printf("mon max\n");

				result = _mm256_fmadd_ps(gt, reg_x, reg_y);

				print_reg(result);

				printf("mon min\n");

				result = _mm256_min_epi32(reg_x, reg_y);

				print_reg(result);


				printf("\n\n");
			} */
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

