#include "timing.h"
#include <time.h>

// in milliseconds
long inittime;
long marktime;

long getTime()
{
	clock_t t = clock();
	return (long)t;
}

void initTimer()
{
    inittime = getTime();
    marktime = getTime();
}

void markTime()
{
    marktime = getTime();
}

long timeSinceMark()
{
    long currenttime = getTime();
    return currenttime - marktime;
}

long timeSinceInit()
{
    long currenttime = getTime();
    return currenttime - inittime;
}