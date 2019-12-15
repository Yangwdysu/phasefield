#pragma once

#include "vector_types.h"
#include <string>
#include <assert.h>
#define __device_builtin__
typedef uchar4 rgb;

typedef float4 vertex; // x, h, z

typedef float4 gridpoint; // h, uh, vh, b

typedef int reflection;
struct __device_builtin__  float7
{
	float a;
	float x0;
	float x1;
	float y0;
	float y1;
	float z0;
	float z1;
};
#define M_PI 3.14159265358979323846
#define M_E 2.71828182845904523536
#define EPSILON 0.000001

#define cudaCheck(x) { cudaError_t err = x;  }
#define synchronCheck {}
