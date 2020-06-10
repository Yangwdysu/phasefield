#pragma once
#pragma once
#include< cuda_runtime.h> 
#include< helper_cuda.h> 
#include< helper_functions.h> 
#include <assert.h>
#include "vector_types.h"
//#include "cudaEssential.h"
namespace WetBrush {

#define INVALID -1
#define EPSILON 0.00001
#define ONE_EPSILON 1.00001

#define RHO1 1000.0f
#define RHO2 10.0f

#define VIS1 10000.0f
#define VIS2 10000.0f

#define MU_INF 100.0f
#define MU_0   0.01f
#define ALPHA  1.0f
#define SCALING 100.0f
#define EXP_N  0.4f

#define MASS_TRESHOLD 0.005f
#define MASS_THESHOLD2 0.5f

#define MAX_PIGMENTS 1	//should be DIM_PIGMENT^3
#define DIM_PIGMENT 1

#define M_PI 3.1415926

#define LAUNCH_KERNEL(name, grid, block, ...)																				\
{																															\
	dim3 gridDims, blockDims;																								\
	computeGridSize3D(grid, block, gridDims, blockDims);																	\
	##name << < gridDims, blockDims>> >(__VA_ARGS__);																		\
	cudaError_t code = cudaDeviceSynchronize();																				\
	if (code != cudaSuccess)																								\
	{																														\
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__); exit(code);					\
	}																														\
}

	struct Coef
	{
		float a;
		float x0;
		float x1;
		float y0;
		float y1;
		float z0;
		float z1;
	};

	__host__ __device__
		void static SeedPigment(float3& pos, int id, int max_num = MAX_PIGMENTS)
	{
		int nn = DIM_PIGMENT*DIM_PIGMENT;
		int k = (id / nn);
		int j = ((id%nn) / DIM_PIGMENT);
		int i = (id%nn) % DIM_PIGMENT;

		float h = 1.0f / DIM_PIGMENT;
		pos = make_float3((i + 0.5f)*h, (j + 0.5f)*h, (k + 0.5f)*h);
	}

	__host__ __device__
		int static GetPigmentIndex(float3 pos)
	{
		float h = 1.0f / DIM_PIGMENT;
		int ix = (int)(pos.x / h);
		int iy = (int)(pos.y / h);
		int iz = (int)(pos.z / h);

		if (ix < 0)	ix = 0;
		if (ix >= DIM_PIGMENT) ix = DIM_PIGMENT - 1;
		if (iy < 0) iy = 0;
		if (iy >= DIM_PIGMENT) iy = DIM_PIGMENT - 1;
		if (iz < 0) iz = 0;
		if (iz >= DIM_PIGMENT) iz = DIM_PIGMENT - 1;

		return ix + iy*DIM_PIGMENT + iz*DIM_PIGMENT*DIM_PIGMENT;
	}

	__host__ __device__
		void static GetPigmentIndex(int3& ids, float3& weight, float3 pos)
	{
		float h = 1.0f / DIM_PIGMENT;
		float fx = pos.x / h;
		float fy = pos.y / h;
		float fz = pos.z / h;
		int ix = floor(fx);
		int iy = floor(fy);
		int iz = floor(fz);

		fx -= ix;	fy -= iy;  fz -= iz;

		if (ix < 0) { ix = 0; fx = 0.0f; }
		if (ix >= DIM_PIGMENT) { ix = DIM_PIGMENT - 1; fx = 1.0f; }
		if (iy < 0) { iy = 0; fy = 0.0f; }
		if (iy >= DIM_PIGMENT) { iy = DIM_PIGMENT - 1; fy = 1.0f; }
		if (iz < 0) { iz = 0; fz = 0.0f; }
		if (iz >= DIM_PIGMENT) { iz = DIM_PIGMENT - 1; fz = 1.0f; }

		fx -= 0.5f;	fy -= 0.5f; fz -= 0.5f;

		ids = make_int3(ix, iy, iz);
		weight = make_float3(fx, fy, fz);
	}

	__host__ __device__
		int static GetPigmentIndex(int i, int j, int k)
	{
		return i + j*DIM_PIGMENT + k*DIM_PIGMENT*DIM_PIGMENT;
	}

	template<typename T>
	class Grid
	{
	public:
		Grid() : nx(0), ny(0), nz(0), nxy(0), data(NULL) {};
		~Grid() {}

		__host__ __device__ inline T operator () (const int i, const int j, const int k) const
		{
			return data[i + j*nx + k*nxy];
		}

		__host__ __device__ inline T& operator () (const int i, const int j, const int k)
		{
			return data[i + j*nx + k*nxy];
		}

		__host__ __device__ inline int Index(const int i, const int j, const int k)
		{
			return i + j*nx + k*nxy;
		}

		__host__ __device__ inline T operator [] (const int id) const
		{
			return data[id];
		}

		__host__ __device__ inline T& operator [] (const int id)
		{
			return data[id];
		}

		__host__ __device__ inline int Size() { return elementCount; }



		/*
		* Only swap the pointer
		*/
		void Swap(Grid<T>& g)
		{
			assert(nx == g.nx && ny == g.ny && nz == g.nz);
			Grid<T> tp = *this;
			*this = g;
			g = tp;
		}

		void CopyFrom(Grid<T>& g)
		{
			checkCudaErrors(cudaMemcpy(data, g.data, elementCount * sizeof(T), cudaMemcpyDeviceToDevice));
		}

		void cudaSetSpace(int _nx, int _ny, int _nz)
		{
			nx = _nx;	ny = _ny;	nz = _nz;	nxy = nx*ny;	elementCount = nxy*nz;
			cudaAllocate();
			cudaClear();
		}

		void cudaClear()
		{
			checkCudaErrors(cudaMemset(data, 0, elementCount * sizeof(T)));
		}

		void cudaAllocate()
		{
			checkCudaErrors(cudaMalloc(&data, elementCount * sizeof(T)));
		}

		void cudaRelease()
		{
			checkCudaErrors(cudaFree(data));
		}


	public:
		int nx;
		int ny;
		int nz;
		int nxy;
		int elementCount;
		T*	data;
	};

	struct PFParameter
	{
		//lame	parameters
		float	gamma;
		//float	h;
		//float	dt;
	};

	typedef Grid<float>	Grid1f;
	typedef Grid<float3> Grid3f;
	typedef Grid<float4> Grid4f;

	typedef Grid<double> Grid1d;
	typedef Grid<double3> Grid3d;

	typedef Grid<bool> Grid1b;

	typedef Grid<Coef> GridCoef;

	typedef Grid<int> Grid1i;

	typedef Grid<int> Grid1u;

	//typedef Grid<uchar4>  rgb;
}