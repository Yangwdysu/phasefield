#include <iostream>
#include <stdio.h>

#include "stdlib.h"
#include "types.h"
//#include "ppm_reader.h"
#include "glwindow.h"
#include "math.h"

int gridsize;
int simulate;
int stepperframe;
int kernelflops;

vertex *landscape;
vertex *wave;
float *waveheights;
rgb *colors;

int argc;
char ** argv;

void update(vertex* wave_vertices, rgb* wave_colors, vertex* landhhs)
{
}

void shutdown()
{
}

void restart()
{
}

int main(int ac, char ** av)
{
    argc = ac;
    argv = av;

	createWindow(argc, argv, 800, 600, gridsize, gridsize, landscape, wave, colors, &update, &restart, &shutdown, stepperframe, kernelflops);

    return 0;
}
