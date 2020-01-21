#ifndef MFD_GRID_H
#define MFD_GRID_H
#include <assert.h>
//#include "MfdMath/Common.h"
//#include "MfdMath/StaticAssert.h"
#include "vector_types.h"
#include "cuda.h"
#include <cuda_runtime.h>

namespace WetBrush {


	template<int dim, typename T>
	class Grid
	{

		//STATIC_ASSERT(dim > 3);

		Grid() { for (int i = 0; i < dim; i++) range[i] = 1; elementCount = 1; }
		Grid(unsigned _range[]) {
			elementCount = 1;
			for (int i = 0; i < dim; i++)
			{
				range[i] = _range[i];
				elementCount *= range[i];
			}
			Allocate();
		}

		int Size() { return elementCount; }

		~Grid() { Release(); }

	public:
		void Allocate() { data = new T[elementCount]; }
		void Release() { if (data != NULL) delete[] data; }

	public:
		unsigned range[dim];
		unsigned int elementCount;
		T* data;
	};

	template<typename T>
	class Grid<2, T>
	{
	public:
		Grid() { nx = 0; ny = 0; elementCount = 0; data = NULL; }
		Grid(int _nx, int _ny) { SetSpace(_nx, _ny); }
		~Grid() { Release(); }

		inline T operator()(const int i, const int j) const
		{
			assert(i >= 0 && i < nx && j >= 0 && j < ny);
			return data[i + j*nx];
		}

		inline T& operator()(const int i, const int j)
		{
			assert(i >= 0 && i < nx && j >= 0 && j < ny);
			return data[i + j*nx];
		}

		inline int Index(const int i, const int j)
		{
			return i + j*nx;
		}

		inline void operator = (Grid<2, T>& g)
		{
			assert(elementCount == g.Size());
			memcpy((void*)data, (void*)g.data, elementCount * sizeof(T));
		}

		void SetSpace(int _nx, int _ny) { nx = _nx; ny = _ny; elementCount = nx*ny; Allocate(); Zero(); }

		void Allocate() { Release(); data = new T[elementCount]; }
		void Release() { if (data != NULL) delete[] data; }

		int Size() { return elementCount; }

		void Zero() { memset((void*)data, 0, elementCount * sizeof(T)); }

		inline T& operator [] (unsigned int id)
		{
			assert(id >= 0 && id < elementCount);
			return data[id];
		}

		inline int Nx() { return nx; }
		inline int Ny() { return ny; }

	public:
		int nx;
		int ny;
		unsigned int elementCount;
		T* data;
	};

	template<typename T>
	class Grid<3, T>
	{
	public:
		Grid() { nx = 0; ny = 0; nz = 0; elementCount = 0; data = NULL; }
		Grid(int _nx, int _ny, int _nz) { SetSpace(_nx, _ny, _nz); }
		~Grid() { /*Release();*/ }

		__host__ __device__  inline T operator()(const int i, const int j, const int k) const
		{
			assert(i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz);
			return data[i + j*nx + k*nxy];
		}

		__host__ __device__  inline T& operator()(const int i, const int j, const int k)
		{
			assert(i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz);
			return data[i + j*nx + k*nxy];
		}

		__host__ __device__  inline int Index(const int i, const int j, const int k)
		{
			return i + j*nx + k*nxy;
		}

		__host__ __device__  inline int GetId(const int i, const int j, const int k) {
			if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz)
				//return INVALID;
			return i + j*nx + k*nxy;
		}

		__host__ __device__  inline T& operator [] (unsigned int id)
		{
			assert(id >= 0 && id < elementCount);
			return data[id];
		}

		__host__ __device__  inline void operator = (Grid<3, T>& g)
		{
			assert(elementCount == g.Size());
			memcpy((void*)data, (void*)g.data, elementCount * sizeof(T));
		}

		void SetSpace(int _nx, int _ny, int _nz) { nx = _nx; ny = _ny; nz = _nz; nxy = nx*ny; elementCount = nx*ny*nz; Allocate(); Zero(); }

		void Allocate() { Release(); data = new T[elementCount]; }
		void Release() { if (data != NULL) delete[] data; }

		__host__ __device__ int Size() { return elementCount; }
		__host__ __device__ void Zero() { memset((void*)data, 0, elementCount * sizeof(T)); }

		inline int Nx() { return nx; }
		inline int Ny() { return ny; }
		inline int Nz() { return nz; }

		__host__ __device__ void Swap(Grid<3, T>& g)
		{
			assert(nx == g.nx && ny == g.ny && nz == g.nz);
			Grid<3, T> tp;
			tp = *this;
			*this = g;
			g = tp;
		}
		//__host__ __device__ void Swap(float* g)
		//{
		//	assert(nx == g.nx && ny == g.ny && nz == g.nz);
		//	float tp = *this;
		//	*this = g;
		//	g = tp;
		//}

		void CopyFromHostToDevice(Grid<3, T>& g)
		{
			(cudaMemcpy(data, g.data, elementCount * sizeof(T), cudaMemcpyHostToDevice));
		}

		void CopyFromDeviceToHost(Grid<3, T>& g)
		{
			(cudaMemcpy(g.data, data, elementCount * sizeof(T), cudaMemcpyDeviceToHost));
		}

		void CopyFromDeviceToDevice(Grid<3, T>& g)
		{
			(cudaMemcpy(data, g.data, elementCount * sizeof(T), cudaMemcpyDeviceToDevice));
		}

		void cudaSetSpace(int _nx, int _ny, int _nz)
		{
			nx = _nx;	ny = _ny;	nz = _nz;	nxy = nx*ny;	elementCount = nxy*nz;
			cudaAllocate();
			cudaClear();
		}

		void cudaClear()
		{
			(cudaMemset(data, 0, elementCount * sizeof(T)));
		}

		void cudaAllocate()
		{
			(cudaMalloc(&data, elementCount * sizeof(T)));
		}

		void cudaRelease()
		{
			(cudaFree(data));
		}
	public:
		int nx;
		int ny;
		int nz;
		int nxy;
		unsigned int elementCount;
		T* data;
	};
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

	struct PFParameter
	{
		//lame	parameters
		float	gamma;
		float	h;
		float	dt;
	};


	template class Grid<2, float>;
	template class Grid<2, double>;
	template class Grid<3, float>;
	typedef Grid<3, float> Grid3f;
	template class Grid<3, double>;

	typedef Grid<2, float> Grid2f;
	typedef Grid<2, double> Grid2d;
	typedef Grid<3, float> Grid3f;
	typedef Grid<3, double> Grid3d;
	typedef Grid<3, Coef> GridCoef;

	//template<typename T>
	//class Gridd
	//{
	//public:
	//	Gridd() : nx(0), ny(0), nz(0), nxy(0), data(NULL) {};
	//	~Gridd() {}

	//	__host__ __device__ inline T operator () (const int i, const int j, const int k) const
	//	{
	//		return data[i + j*nx + k*nxy];
	//	}

	//	__host__ __device__ inline T& operator () (const int i, const int j, const int k)
	//	{
	//		return data[i + j*nx + k*nxy];
	//	}

	//	__host__ __device__ inline int Index(const int i, const int j, const int k)
	//	{
	//		return i + j*nx + k*nxy;
	//	}

	//	__host__ __device__ inline T operator [] (const int id) const
	//	{
	//		return data[id];
	//	}

	//	__host__ __device__ inline T& operator [] (const int id)
	//	{
	//		return data[id];
	//	}
	//	//__host__ __device__ inline T LengthSq() const
	//	//{
	//	//	return x*x + y*y + z*z;
	//	//}

	//	__host__ __device__ inline T Length() const
	//	{
	//		return sqrt(LengthSq());
	//	}
	//	__host__ __device__ inline int Size() { return elementCount; }

	//	/*
	//	* Only swap the pointer
	//	*/
	//	//void Swap(Gridd<T>& g)
	//	//{
	//	//	assert(nx == g.nx && ny == g.ny && nz == g.nz);
	//	//	Gridd<T> tp = *this;
	//	//	*this = g;
	//	//	g = tp;
	//	//}

	//	void CopyFrom(Gridd<T>& g)
	//	{
	//		checkCudaErrors(cudaMemcpy(data, g.data, elementCount * sizeof(T), cudaMemcpyHostToDevice));
	//	}

	//	void cudaSetSpace(int _nx, int _ny, int _nz)
	//	{
	//		nx = _nx;	ny = _ny;	nz = _nz;	nxy = nx*ny;	elementCount = nxy*nz;
	//		Allocate();
	//		Clear();
	//	}

	//	void Clear()
	//	{
	//		checkCudaErrors(cudaMemset(data, 0, elementCount * sizeof(T)));
	//	}

	//	void Allocate()
	//	{
	//		checkCudaErrors(cudaMalloc(&data, elementCount * sizeof(T)));
	//	}

	//	void Release()
	//	{
	//		checkCudaErrors(cudaFree(data));
	//	}


	//public:
	//	int nx;
	//	int ny;
	//	int nz;
	//	int nxy;
	//	int elementCount;
	//	T*	data;
	//};

	////struct PFParameter
	////{
	////	//lame	parameters
	////	float	gamma;
	////	float	h;
	////	float	dt;
	////};

	//typedef Gridd<float3>Gridv3f;



}

#endif