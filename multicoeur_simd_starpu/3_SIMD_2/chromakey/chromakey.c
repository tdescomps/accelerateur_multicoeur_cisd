#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

/* Size in bytes of an SIMD register */
#define REG_BYTES (sizeof(__m256i))

/* Number of elements in an SIMD register */
#define REG_NB_ELEMENTS (REG_BYTES / sizeof(uint8_t))

void usage (void)
{
	fprintf(stderr, "chromakey [--fg <FOREGROUND PPM IMAGE>] [--bg <BACKGREOUND PPM IMAGE>] [--output <OUTPUT IMAGE>]\n");
	exit(EXIT_FAILURE);
}

struct s_ppm_image
{
	int width;
	int height;
	int max_value;
	uint8_t *red;
	uint8_t *green;
	uint8_t *blue;
};

struct s_ppm_image *read_ppm_file(const char *filename)
{
	FILE *f;

	f = fopen(filename, "r");
	if (f == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	int state = 0;

	int width = 0;
	int height = 0;
	int max_value = 0;
	uint8_t *red = NULL;
	uint8_t *green = NULL;
	uint8_t *blue = NULL;
	int nb_pixels = 0;

	while (state < 3 || ((state >= 3) && (state < 6) && (nb_pixels < (width*height))))
	{
		assert(state >=0);
		const int max_line_len = 256;
		char line[max_line_len];

		if (fgets(line, max_line_len, f) == NULL)
		{
			perror("fgets");
			exit(EXIT_FAILURE);
		}

		if (state == 0)
		{
			printf("mark = [%s]\n", line);
			if (strcmp(line, "P3\n") != 0)
			{
				fprintf(stderr, "invalid file mark\n");
				exit(EXIT_FAILURE);
			}

			state = 1;
		}
		else
		{
			if (line[0] == '#')
				continue;
			if (state == 1)
			{
				int n = sscanf(line, "%d %d", &width, &height);
				if (n < 2)
				{
					fprintf(stderr, "invalid width or height\n");
					exit(EXIT_FAILURE);
				}
				printf("width = %d\n", width);
				printf("height = %d\n", height);
				state = 2;
			}
			else if (state == 2)
			{
				int n = sscanf(line, "%d", &max_value);
				if (n < 1)
				{
					fprintf(stderr, "invalid max_value\n");
					exit(EXIT_FAILURE);
				}
				printf("max_value = %d\n", max_value);

				/* assume max component value is 255 */
				assert(max_value == 255);

				red = aligned_alloc(REG_BYTES, width*height*sizeof(*red));
				assert(red != NULL);

				green = aligned_alloc(REG_BYTES, width*height*sizeof(*green));
				assert(green != NULL);

				blue = aligned_alloc(REG_BYTES, width*height*sizeof(*blue));
				assert(blue != NULL);
				state = 3;
			}
			else if (state == 3)
			{
				int red_value;
				int n = sscanf(line, "%d", &red_value);
				if (n < 1)
				{
					fprintf(stderr, "invalid max_value\n");
					exit(EXIT_FAILURE);
				}
				red[nb_pixels] = red_value;
				state = 4;
			}
			else if (state == 4)
			{
				int green_value;
				int n = sscanf(line, "%d", &green_value);
				if (n < 1)
				{
					fprintf(stderr, "invalid max_value\n");
					exit(EXIT_FAILURE);
				}
				green[nb_pixels] = green_value;
				state = 5;
			}
			else if (state == 5)
			{
				int blue_value;
				int n = sscanf(line, "%d", &blue_value);
				if (n < 1)
				{
					fprintf(stderr, "invalid max_value\n");
					exit(EXIT_FAILURE);
				}
				blue[nb_pixels] = blue_value;

				nb_pixels++;

				if (nb_pixels >= (width * height))
				{
					state = 6;
					break;
				}

				state = 3;
			}
		}
	}
	assert(state == 6);
	fclose(f);

	struct s_ppm_image *ppm_image = calloc(1, sizeof(*ppm_image));
	ppm_image->width = width;
	ppm_image->height = height;
	ppm_image->max_value = max_value;
	ppm_image->red = red;
	ppm_image->green = green;
	ppm_image->blue = blue;

	return ppm_image;
}

void write_ppm_image(struct s_ppm_image* ppm_image, const char *filename)
{
	FILE *f;

	f = fopen(filename, "w");
	if (f == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	fprintf(f, "P3\n");
	fprintf(f, "# chromakey\n");
	fprintf(f, "%d %d\n", ppm_image->width, ppm_image->height);
	fprintf(f, "%d\n", ppm_image->max_value);

	int i;
	for (i=0; i<ppm_image->width * ppm_image->height; i++)
	{
		fprintf(f, "%d\n", ppm_image->red[i]);
		fprintf(f, "%d\n", ppm_image->green[i]);
		fprintf(f, "%d\n", ppm_image->blue[i]);
	}
	fclose(f);
}

void free_ppm_image(struct s_ppm_image* ppm_image)
{
	free(ppm_image->red);
	free(ppm_image->green);
	free(ppm_image->blue);
	memset(ppm_image, 0, sizeof(*ppm_image));
	free(ppm_image);
}

void apply_chromakey(struct s_ppm_image *fg,struct s_ppm_image *bg)
{
	assert(fg->width == bg->width);
	assert(fg->height == bg->height);
	const int len = fg->width * fg->height;

	const uint8_t chroma_key_red_value   = 0x00;
	const uint8_t chroma_key_green_value = 0x00;
	const uint8_t chroma_key_blue_value  = 0xff;

	/*** ------------------------------------ ***/
	/*** add the function implementation here ***/
	/*** ------------------------------------ ***/
}

int main(int argc, char *argv[])
{
	char *fg_filename = NULL;
	struct s_ppm_image *fg_image = NULL;
	char *bg_filename = NULL;
	struct s_ppm_image *bg_image = NULL;
	char *output_filename = NULL;

	{
		int i = 1;
		while (i < argc)
		{
			char *arg = argv[i];
			if (strcmp(arg, "--fg") == 0)
			{
				if (fg_filename != NULL)
				{
					usage();
				}
				i++;
				fg_filename = argv[i];
			}
			else if (strcmp(arg, "--bg") == 0)
			{
				if (bg_filename != NULL)
				{
					usage();
				}
				i++;
				bg_filename = argv[i];
			}
			else if (strcmp(arg, "--output") == 0)
			{
				if (output_filename != NULL)
				{
					usage();
				}
				i++;
				output_filename = argv[i];
			}
			else
			{
				usage();
				exit(EXIT_FAILURE);
			}
			i++;
		}
	}

	if (fg_filename == NULL || bg_filename == NULL)
		usage();

	if (strcmp(fg_filename, output_filename) == 0)
	{
		fprintf(stderr, "output filename identical to foreground filename\n");
		exit(EXIT_FAILURE);
	}

	if (strcmp(bg_filename, output_filename) == 0)
	{
		fprintf(stderr, "output filename identical to background filename\n");
		exit(EXIT_FAILURE);
	}

	fg_image = read_ppm_file(fg_filename);
	bg_image = read_ppm_file(bg_filename);

	apply_chromakey(fg_image, bg_image);

	write_ppm_image(fg_image, output_filename);

	free_ppm_image(fg_image);
	free_ppm_image(bg_image);

	return 0;
}
