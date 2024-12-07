#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

#define win_width  512
#define win_height 512
#define module_threshold (1.0e10)

#define loop_max_threshold 255

static float view_width=2;
static float view_height=2;
static float orig_x=-0.5;
static float orig_y=-0.0;

static int loop_threshold=1;

static int colormap[loop_max_threshold];

/* XWindow objects*/
static Display *display = (Display *)NULL;
static Window root_window;
static int screen = -1;
static int depth = -1;
static Visual *visual = (Visual *)NULL;
static Window window ;
static XSetWindowAttributes win_attr;
static GC win_gc;
static XImage *blit_image = NULL;

static int mand_array[win_width * win_height];

static void
init_colormap(void) {
	for (int i=0; i<loop_max_threshold; i++) {
		colormap[i] = !!(i&1) * 0xff + !!(i&2) * 0xff00 + !!(i&4)*0xff0000;
	}
}

static void
update_mand_array(void) {
	const float scale_x = view_width / (float)win_width;
	const float scale_y = view_height / (float)win_height;
	const float display_offset_x = win_width / 2;
	const float display_offset_y = win_height / 2;

	for (int y = 0; y < win_height ; y++) {
		for (int x = 0; x < win_width ; x++) {
			const float ca = scale_x * (x-display_offset_x) + orig_x;
			const float cb = scale_y * (y-display_offset_y) + orig_y;
			float xa = ca;
			float xb = cb;
			int nb_iter = 0;

			do {
				float new_xa = xa*xa - xb*xb + ca;
				float new_xb = 2 * xa*xb + cb;
				xa = new_xa;
				xb = new_xb;
				nb_iter++;
			} while ((nb_iter < loop_threshold) && ((xa*xa + xb*xb) < module_threshold));

			int pixel = (nb_iter < loop_threshold)?colormap[nb_iter]:0;
			mand_array[win_width*y+x] = pixel;
		}
	}
	loop_threshold = loop_threshold+1;
	if (loop_threshold > loop_max_threshold) {
		loop_threshold = 1;
	}
}

static void
display_mand_array(void) {
	int err;
	if ((err = XPutImage(display, window, win_gc, blit_image, 0, 0, 0, 0, win_width, win_height))) {
		fprintf(stderr, "XPutImage failed\n");
		exit(1);
	}
	XSync(display, False);
}

static void
display_f(void) {
	update_mand_array();
	display_mand_array();
}

int
main(int argc, char ** argv) {
	init_colormap();
	display = XOpenDisplay(0);
	if (display == (Display *)NULL) {
		(void)fprintf(stderr, "XOpenDisplay failed\n");
		exit(1);
	}

	root_window = DefaultRootWindow(display);
	screen = DefaultScreen(display);
	depth = DefaultDepth(display, screen);

	//visual = DefaultVisual(display, screen);
	{
		/* find a 24-bit visual */
		XVisualInfo visual_24bit_info;
		if (!XMatchVisualInfo(display, screen, 24, TrueColor, &visual_24bit_info)) {
			fprintf(stderr, "24-bit visual not found\n");
			exit(1);
		}
		visual = visual_24bit_info.visual;
	}

	/* Init window */
	memset(&win_attr, 0, sizeof(XSetWindowAttributes));
	win_attr.background_pixel = 0;
	win_attr.border_pixel = 0;
	win_attr.colormap = XCreateColormap(display, root_window, visual, AllocNone);
	window = XCreateWindow(display,
			root_window,
			0, 0,
			win_width, win_height,
			0,
			depth,
			InputOutput,
			visual,
			CWBackPixel|CWBorderPixel|CWColormap, &win_attr);

	/* Set window title */
	XStoreName(display, window, "X11 Window");

	/* Initialize which set of events we want to handle */
	XSelectInput(display, window, KeyPressMask);

	/* Show the window on the display */
	XMapWindow (display, window);

	/* Sync XWindow command queue */
	XSync(display, True) ;

	/* Create a graphic context for the window */
	win_gc = XCreateGC(display, window, 0L, 0);

	/* Create a X11 wrapper for our mand_array */
	if (!(blit_image = XCreateImage(display, NULL, 24, ZPixmap, 0, (char*)mand_array, win_width, win_height, 32, 0))) {
		fprintf(stderr, "XCreateImage failed\n");
		exit(1);
	}

	/* Update and display mand array */
	display_f();

	for (;;) {
		XEvent event;

		if (XCheckTypedEvent(display, KeyPress, &event) == True) {
			switch (event.type) {
				case KeyPress:
					exit(0);
					break;
				default:
					fprintf(stderr, "unhandled X11 event\n");
					exit(1);
			}
		}
		display_f();
	}

	return 0;
}

