#if __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/glew.h>
#endif
#include "cuda_helper_math.h"
#include"cuda_runtime.h"
#include <cuda_gl_interop.h> 
#include<device_launch_parameters.h>
#include <cufft.h>
#include "gl_utilities.h"
#include <time.h>
#include "PhaseField.h"
#include"Ocean.h"
#include <iostream>

namespace WetBrush {
#define BLOCK_SIZE 16
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define BLOCKSIZE_Z 16
#ifdef NDEBUG
#define cuSynchronize() {}
#else
#define cuSynchronize()	{						\
		char str[200];							\
		cudaDeviceSynchronize();				\
		cudaError_t err = cudaGetLastError();	\
		if (err != cudaSuccess)					\
		{										\
			sprintf(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);		\
			throw std::runtime_error(std::string(str));																\
		}																											\
	}

#endif
#define grid3Dwrite(array, x, y, z,value) array[(z)*nx*ny+(y)*ny+x] = value
#define grid3Dread(array, x, y, z) array[(z)*nx*ny+(y)*nx+x]
	//float m_virtualGridSize = 0.1;
	float samplingDistance = 0.02f;
	/*
	date:2019/11/14
	author:@wdy
	describe:Data copy on GPU
	*/

	//__global__ void K_CopyGData(Grid1f dst, Grid1f src, int nx, int ny, int nz)
	//{
	//	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//	int k = blockIdx.z * blockDim.z + threadIdx.z;

	//	//int index;
	//	if (i >= nx) return;
	//	if (j >= ny) return;
	//	if (k >= nz) return;
	//	int index = i + j*nx + k*nx*ny;
	//	//dst.x = src.x;
	//	//dst.y = src.y;
	//	//dst.z = src.z;
	//	//dst.w = src.w;
	//	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	//	{
	//		dst(i, j, k) = src(i, j, k);

	//	}

	//	//float gp;
	//	//gp = 0.0;;
	//	//gp.y = 0.0f;
	//	//gp.z = 0.0f;
	//	//gp.w = 0.0f;
	//	//grid3Dwrite(dst, i, j, k, gp);
	//}


	__global__ void K_CopyGData(Grid1f dst, Grid1f src, int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= src.nx) return;
		if (j >= src.ny) return;
		if (k >= src.nz) return;
		dst(i, j, k) = src(i, j, k);
		//}
	}
	__global__ void K_CopyFData(float* dst, float* src, int nx, int ny, int nz)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		//int index;
		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;
		int index = i + j*nx + k*nx*ny;
		dst[index] = src[index];

	}
	/*******************************************************************************/
	/***********************************Init_condition******************************/
	/*******************************************************************************/
	PhaseField::PhaseField()
	{
		//m_patch_length = patchLength;
		//m_realGridSize = patchLength / size;

		sizem = 100;

		simulatedRegionLenght = sizem;
		simulatedRegionWidth = sizem;
		simulatedRegionHeight = sizem;

		AllocateMemoery(sizem, sizem, sizem);


	}

	PhaseField::~PhaseField(void)
	{
	}

	void PhaseField::initialize()
	{

		initRegion();
	}



	__global__ void C_InitDynamicRegion(float4* moveRegion, rgb* color, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		//printf("%d", nz);

		int index;
		float m_virtualGridSize = 0.1f;

		if (i >= 0 && i < nx  && j >= 0 && j < ny && k >= 0 && k < nz)
		{
			index = i + j*nx + k*nx*ny;

			float4 vij;
			vij.x = (float)i*m_virtualGridSize;
			vij.y = (float)0;
			vij.z = (float)j*m_virtualGridSize;
			vij.w = (float)1;

			moveRegion[index] = vij;
			//color[index] = make_uchar4(0, 120, 40, 220);
			//printf("%f", moveRegion[index].x);
		}
	}

	__global__ void C_InitPosition(Grid3f position, rgb* color, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		int index;
		float samplingDistance = 0.1f;

		if (i >= 0 && i < nx  && j >= 0 && j < ny && k >= 0 && k < nz)
		{
			index = i + j*nx + k*nx*ny;
			float x = i * samplingDistance;
			float y = j * samplingDistance;
			float z = k * samplingDistance;
			//position(i, j, k) = make_float3(x, y, z);
			position(i, j, k).x = x;
			position(i, j, k).y = y;
			position(i, j, k).z = z;
			//color[index] = make_uchar4(120, 120, 120, 220);
		}
	}


	__global__ void C_PhaseField(Grid1f phasefield0, int nx, int ny, int nz)
	{
		int index;
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			index = i + j*nx + k*nx*ny;
			if (i > 5 && i < 35 && j > 20 && j < 80 && k>5 && k < 35)
			{
				//phasefield[index] = 1.0f;
				phasefield0(i, j, k) = 1.0f;
			}
			else
			{
				//phasefield[index] = 0.0f;
				phasefield0(i, j, k) = 0.0f;
			}
		}
	}


	__global__ void C_PhaseChange(Grid1f phasefield0, rgb* color, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		int index;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			index = i + j*nx + k*nx*ny;

			float phiijk = phasefield0(i, j, k);
			phiijk = clamp(phiijk, float(0), float(1));

			color[index] = make_uchar4(0, 120, 120, phasefield0(i, j, k) * 255);
		}
	}


	__global__ void C_PhaseChange1(Grid1f phasefield, rgb* color, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		int index;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			index = i + j*nx + k*nx*ny;
			if (phasefield(i, j, k) == 0.0)
			{
				color[index] = make_uchar4(5, 255, 255, 0);
			}
			if (phasefield(i, j, k) == 1.0)
			{
				color[index] = make_uchar4(0, 0, 120, 220);
			}
			//if (phasefield(i, j, k) > 0.0 && phasefield(i, j, k)< 1.0)
			//{
			//	color[index] = make_uchar4(0, 120, 120, phasefield(i, j, k) * 255);
			//}
		}
	}

	void PhaseField::initRegion()
	{
		cudaError_t error;
		int extNx = simulatedRegionLenght;
		int extNy = simulatedRegionWidth;
		int extNz = simulatedRegionHeight;


		//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);



		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		C_InitPosition << < dimGrid, dimBlock >> > (m_cuda_position, m_cuda_color, nx, ny, nz);
		synchronCheck;

		C_PhaseField << <dimGrid, dimBlock >> > (m_cuda_phasefield0, nx, ny, nz);
		synchronCheck;

		error = cudaThreadSynchronize();

	}




	__global__ void C_moveSimulationRegion(float4* moveRegion, int nx, int ny, int nz, int dx, int dy)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		int index;

		if (i >= 0 && i < nx  && j >= 0 && j < ny && k >= 0 && k < nz)
		{

			index = i + j*nx + k*nx*ny;

			moveRegion[index].x += dx*0.01;
			moveRegion[index].y += 0;
			moveRegion[index].z += dy*0.01;
			moveRegion[index].w += 0;

		}
	}

	void PhaseField::moveSimulationRegion(int dx, int dy)
	{
		//int eNx = simulatedRegionLenght;
		//int eNy = simulatedRegionWidth;
		//int eNz = simulatedRegionHeight;
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((simulatedRegionLenght + dimBlock.x - 1) / dimBlock.x, (simulatedRegionWidth + dimBlock.y - 1) / dimBlock.y, (simulatedRegionHeight + dimBlock.z - 1) / dimBlock.z);
		C_moveSimulationRegion << < dimGrid, dimBlock >> > (m_cuda_SimulationRegion, simulatedRegionLenght, simulatedRegionWidth, simulatedRegionHeight, dx, dy);
		synchronCheck;

		//m_simulatedOriginX += dx;
		//m_simulatedOriginY += dy;
		//cudaGraphicsUnmapResources(1, &SimulationRegion_resource, 0);
		//cudaGraphicsUnmapResources(1, &SimulationRegionColor_resource, 0);
		//cudaFree(m_cuda_SimulationRegion);

	}



	/*
	date:2019/12/07
	author:@wdy
	describe:allocate memoery for all variable
	*/
	void PhaseField::AllocateMemoery(int _nx, int _ny, int _nz)
	{

		m_simulatedOriginX = 0;
		m_simulatedOriginY = 0;

		nx = _nx;//计算区域-流体部分
		ny = _ny;
		nz = _nz;


		simulationRegitionSize = simulatedRegionLenght*simulatedRegionWidth*simulatedRegionHeight;
		simulationSize = nx*ny*nz;
		//for phasefield equation
		//仿真区域位置及颜色
		cudaCheck(cudaMalloc(&m_cuda_SimulationRegion, simulationRegitionSize * sizeof(float4)));
		cudaCheck(cudaMalloc(&m_cuda_SimulationRegionColor, simulationRegitionSize * sizeof(rgb)));
		//流体部分
		//cudaCheck(cudaMalloc(&m_cuda_position, simulationSize * sizeof(float4)));
		cudaCheck(cudaMalloc(&m_cuda_color, simulationSize * sizeof(uchar4)));
		//cudaCheck(cudaMalloc(&m_cuda_phasefield, simulationSize * sizeof(float)));
		//cudaCheck(cudaMalloc(&m_cuda_phasefield0, simulationSize * sizeof(float)));
		//cudaCheck(cudaMalloc(&m_cuda_Velu, (nx+1)*ny*nz * sizeof(float)));
		//cudaCheck(cudaMalloc(&m_cuda_Velv, nx*(ny+1)*nz * sizeof(float)));
		//cudaCheck(cudaMalloc(&m_cuda_Velw, nx*ny*(nz+1) * sizeof(float)));
		//cudaCheck(cudaMalloc(&m_cuda_Veluc, nx*ny*nz * sizeof(float3)));
		//cudaCheck(cudaMalloc(&m_cuda_Veluc0, nx*ny*nz * sizeof(float3)));
		//cudaCheck(cudaMalloc(&m_cuda_Veluc1, nx*ny*nz * sizeof(float3)));
		cudaCheck(cudaMalloc(&max_velu, 1 * sizeof(float)));
		cfl = (float*)malloc(1 * sizeof(float));

		m_cuda_position.cudaSetSpace(nx, ny, nz);
		//m_cuda_color.cudaSetSpace(nx, ny, nz);
		//m_cuda_SimulationRegionColor.cudaSetSpace(nx, ny, nz);
		//m_cuda_SimulationRegion.cudaSetSpace(nx, ny, nz);
		m_cuda_phasefield0.cudaSetSpace(nx, ny, nz);
		m_cuda_phasefield.cudaSetSpace(nx, ny, nz);
		//for N-S equation
		m_cuda_Velu.cudaSetSpace(nx + 1, ny, nz);
		m_cuda_Velv.cudaSetSpace(nx, ny + 1, nz);
		m_cuda_Velw.cudaSetSpace(nx, ny, nz + 1);
		m_cuda_Veluc.cudaSetSpace(nx, ny, nz);
		m_cuda_Veluc0.cudaSetSpace(nx, ny, nz);
		m_cuda_CoefMatrix.cudaSetSpace(nx, ny, nz);
		m_cuda_RHS.cudaSetSpace(nx, ny, nz);
		m_cuda_Pressure.cudaSetSpace(nx, ny, nz);
		m_cuda_BufPressure.cudaSetSpace(nx, ny, nz);




		size_t size;





		//粒子位置、相场及颜色
		glGenBuffers(1, &Initpos_bufferObj);
		glBindBuffer(GL_ARRAY_BUFFER, Initpos_bufferObj);
		glBufferData(GL_ARRAY_BUFFER, simulationSize * sizeof(float3), NULL, GL_DYNAMIC_COPY);
		cudaGraphicsGLRegisterBuffer(&Initpos_resource, Initpos_bufferObj, cudaGraphicsMapFlagsWriteDiscard);
		cudaGraphicsMapResources(1, &Initpos_resource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&m_cuda_position.data, &size, Initpos_resource);


		glGenBuffers(1, &Color_bufferObj);
		glBindBuffer(GL_ARRAY_BUFFER, Color_bufferObj);
		glBufferData(GL_ARRAY_BUFFER, simulationSize * sizeof(uchar4), NULL, GL_STATIC_DRAW);
		cudaGraphicsGLRegisterBuffer(&Color_resource, Color_bufferObj, cudaGraphicsMapFlagsWriteDiscard);
		cudaGraphicsMapResources(1, &Color_resource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&m_cuda_color, &size, Color_resource);


	}




	__global__ void C_CFL(float* max_velu, float samplingDistance, Grid1f Velu_x, Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		int k = threadIdx.z + blockIdx.z * blockDim.z;
		//float max_velu;
		float maxvel = 0.0f;
		if (i >= 0 && i < Velu_x.Size())
		{
			maxvel = max(maxvel, abs(Velu_x[i]));
		}
		if (j >= 0 && j < Velv_y.Size())
		{
			maxvel = max(maxvel, abs(Velv_y[j]));
		}
		if (k >= 0 && k < Velw_z.Size())
		{
			maxvel = max(maxvel, abs(Velw_z[k]));
		}
		if (maxvel < EPSILON)
			maxvel = 1.0f;
		max_velu[0] = samplingDistance / maxvel;
	}
	/*******************************************************************************/
	/***********************************Program entry*******************************/
	/*******************************************************************************/

	static float t = -0.01f;
	void PhaseField::animate(float dt)
	{
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);

		clock_t total_start = clock();
		clock_t t_start = clock();
		clock_t t_end;
		float elipse = 0.0f;
		float dx = 1.0f / nx;
		float T = 4.0f;
		t_end = clock();
		cout << "Solving Pressure Costs: " << t_end - t_start << endl;
		t_start = clock();
		while (elipse < dt) {
			C_CFL << <dimGrid, dimBlock >> > (max_velu, samplingDistance, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz);
			cuSynchronize();
			//CFL(max_velu);
			cudaMemcpy(cfl, max_velu, 1 * sizeof(float), cudaMemcpyDeviceToHost);
			float substep = cfl[0];
			//float substep = 0.001f;// CFL();
			if (elipse + substep > dt)
			{
				substep = dt - elipse;
			}
			cout << "*********Substep: " << substep << " *********" << endl;
			t_start = clock();

			//N-S solver
			NS_solver(substep);
			t_start = clock();

			////m_cuda_phasefield0 = m_cuda_phasefield;
		/*	K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield, m_cuda_phasefield0, nx, ny, nz);*/
			////PF solver
			PF_solver(substep);
			//C_Velecity << <dimGrid, dimBlock >> > (m_cuda_position, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, substep);

			//synchronCheck;


			//cudaGraphicsUnmapResources(1, &Velu_resource, 0);
			//cudaGraphicsUnmapResources(1, &Velv_resource, 0);
			//cudaGraphicsUnmapResources(1, &Velw_resource, 0);

			t_end = clock();
			cout << "Advect Time: " << t_end - t_start << endl;
			elipse += substep;
		}
		C_PhaseChange << <dimGrid, dimBlock >> > (m_cuda_phasefield, m_cuda_color, nx, ny, nz);
		//printf("%f"，m_cuda_phasefield0.data);

		cout << dt << endl;
		t += dt;
		clock_t total_end = clock();
		cout << "Total Cost " << total_end - total_start << " million seconds!" << endl;
		//cudaGraphicsUnmapResources(1, &Initpos_resource, 0);
		//if (simItor*dt > 4.01f)
		//{
		//	exit(0);
		//}

	}


	/*
	date:2019/11/14
	author:@wdy
	describe:CFL condition
	*/
	//#define INNERINDEX(m,n,l) (m-1)*(ny-2)*(nz-2)+(n-1)*(nz-2)+l-1
	//float PhaseField::CFL(float max_velu)
	//{
	//	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	//	dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
	//
	//	C_CFL << <dimGrid, dimBlock >> > (max_velu, samplingDistance, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz);
	//	cuSynchronize();
	//	//return max_velu;
	//}





	/*******************************************************************************/
	/********************************PhaseField_Solver******************************/
	/*******************************************************************************/

	__global__ void P_AdvectWENO1rd(Grid1f d, Grid1f d0, Grid1f Velu, Grid1f Velv, Grid1f Velw, int nx, int ny, int nz, float dt)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int index;
		int index0;
		int index1;
		float samplingDistance = 0.005f;
		float invh = 1.0f / samplingDistance;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{

			float u_mid;
			float c_mid;
			int ix0, iy0, iz0;
			int ix1, iy1, iz1;
			float dc;

			ix0 = i;   iy0 = j; iz0 = k;
			ix1 = i + 1; iy1 = j; iz1 = k;
			if (ix1 < nx - 1)
			{
				u_mid = Velu(i + 1, j, k);
				if (u_mid > 0.0f)
				{
					c_mid = d0(ix0, iy0, iz0);
				}
				else
				{
					c_mid = d0(ix1, iy1, iz1);
				}
				dc = dt*invh*c_mid*u_mid;
				//d(ix0, iy0, iz0) -= dc;
				//d(ix1, iy1, iz1) += dc;
				atomicAdd(&d(ix0, iy0, iz0), -dc);
				atomicAdd(&d(ix1, iy1, iz1), dc);

			}


			//j and j+1
			ix0 = i; iy0 = j;   iz0 = k;
			ix1 = i; iy1 = j + 1; iz1 = k;
			if (iy1 < ny - 1)
			{
				u_mid = Velv(i, j + 1, k);
				if (u_mid > 0.0f)
				{
					c_mid = d0(ix0, iy0, iz0);
				}
				else
				{
					c_mid = d0(ix1, iy1, iz1);
				}
				dc = dt*invh*c_mid*u_mid;
				//d(ix0, iy0, iz0) -= dc;
				//d(ix1, iy1, iz1) += dc;
				atomicAdd(&d(ix0, iy0, iz0), -dc);
				atomicAdd(&d(ix1, iy1, iz1), dc);
			}

			ix0 = i; iy0 = j;   iz0 = k;
			ix1 = i; iy1 = j; iz1 = k + 1;
			if (iz1 < nz - 1)
			{
				u_mid = Velw(i, j, k + 1);
				if (u_mid > 0.0f)
				{
					c_mid = d0(ix0, iy0, iz0);
				}
				else
				{
					c_mid = d0(ix1, iy1, iz1);
				}
				dc = dt*invh*c_mid*u_mid;
				//d(ix0, iy0, iz0) -= dc;
				//d(ix1, iy1, iz1) += dc;

				atomicAdd(&d(ix0, iy0, iz0), -dc);
				atomicAdd(&d(ix1, iy1, iz1), dc);
			}




		}
	}


	/*-----------------------------------------------------------------------------*/
	/*-----------------------------------AdvectForward---------------------------- */
	/*-----------------------------------------------------------------------------*/
	__global__ void Kenel_AdvectForward(Grid1f d0, Grid1f d, Grid1f Velu, Grid1f Velv, Grid1f Velw, int nx, int ny, int nz, float dt)
	{
		float h = 0.005f;
		float fx, fy, fz;
		int  ix, iy, iz;
		float w000, w100, w010, w001, w111, w011, w101, w110;
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			int idx = i + j*nx + k*nx*ny;
			fx = i + dt*Velu(i, j, k) / h;
			fy = j + dt*Velv(i, j, k) / h;
			fz = k + dt*Velw(i, j, k) / h;
			if (fx < 1) { fx = 1; }
			if (fx > nx - 2) { fx = nx - 2; }
			if (fy < 1) { fy = 1; }
			if (fy > ny - 2) { fy = ny - 2; }
			if (fz < 1) { fz = 1; }
			if (fz > nz - 2) { fz = nz - 2; }

			ix = (int)fx;
			iy = (int)fy;
			iz = (int)fz;
			fx -= ix;
			fy -= iy;
			fz -= iz;
			float& val = d0[idx];
			//float& val = phi0(i,j,k);
			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			w100 = fx * (1.0f - fy)*(1.0f - fz);
			w010 = (1.0f - fx)*fy *(1.0f - fz);
			w001 = (1.0f - fx)*(1.0f - fy)*fz;
			w111 = fx * fy * fz;
			w011 = (1.0f - fx)*fy * fz;
			w101 = fx * (1.0f - fy)*fz;
			w110 = fx * fy *(1.0f - fz);
			//原子操作
			atomicAdd(&d(ix, iy, iz), val * w000);
			atomicAdd(&d(ix + 1, iy, iz), val * w100);
			atomicAdd(&d(ix, iy + 1, iz), val * w010);
			atomicAdd(&d(ix, iy, iz + 1), val * w001);

			atomicAdd(&d(ix + 1, iy + 1, iz + 1), val * w111);
			atomicAdd(&d(ix, iy + 1, iz + 1), val * w011);
			atomicAdd(&d(ix + 1, iy, iz + 1), val * w101);
			atomicAdd(&d(ix + 1, iy + 1, iz), val * w110);

		}
	}

	/*
	date:2019/11/14
	author:@wdy
	describe:phasefield solver
	*/


	void PhaseField::PF_solver(float substep)
	{
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield, m_cuda_phasefield0, nx, ny, nz);
		PF_setScalarFieldBoundary(true);
		//Kenel_AdvectForward << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, substep);
		//adection
		P_AdvectWENO1rd << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, substep);
		cuSynchronize();
		//for (size_t i = 0; i < m_cuda_phasefield0.Size(); i++)
		//{
		//	if (m_cuda_phasefield0[i]!=0.0)
		//	{
		//		printf("%f", m_cuda_phasefield0[i]);
		//	}
		//	
		//}
		
		//cout << "成功！！" <<endl;
	}
	__global__ void SetScalarFieldBoundary_x(Grid1f field, float s, int nx, int ny, int nz)
	{

		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			field(0, j, k) = s*field(1, j, k);
			field(nx - 1, j, k) = s*field(nx - 2, j, k);
		}
	}

	__global__ void SetScalarFieldBoundary_y(Grid1f field, float s, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= 1 && i < nx - 1 && k >= 1 && k < nz - 1)
		{
			field(i, 0, k) = s*field(i, 1, k);
			field(i, ny - 1, k) = s*field(i, ny - 2, k);
		}
	}

	__global__ void SetScalarFieldBoundary_z(Grid1f field, float s, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
		{
			field(i, j, 0) = s*field(i, j, 1);
			field(i, j, nz - 1) = s*field(i, j, nz - 2);
		}
	}

	__global__ void SetScalarFieldBoundary_yz(Grid1f field,int nx, int ny, int nz)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= 0 && i < nx)
		{
			field(i, 0, 0) = 0.5f*(field(i, 1, 0) + field(i, 0, 1));
			field(i, ny - 1, 0) = 0.5f*(field(i, ny - 2, 0) + field(i, ny - 1, 1));
			field(i, 0, nz - 1) = 0.5f*(field(i, 1, nz - 1) + field(i, 0, nz - 2));
			field(i, ny - 1, nz - 1) = 0.5f*(field(i, ny - 1, nz - 2) + field(i, ny - 2, nz - 1));
		}
	}


	__global__ void SetScalarFieldBoundary_xz(Grid1f field, int nx, int ny, int nz)
	{
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (j >= 1 && j < ny - 1)
		{
			field(0, j, 0) = 0.5f*(field(1, j, 0) + field(0, j, 1));
			field(0, j, nz - 1) = 0.5f*(field(1, j, nz - 1) + field(0, j, nz - 2));
			field(nx - 1, j, 0) = 0.5f*(field(nx - 2, j, 0) + field(nx - 2, j, 1));
			field(nx - 1, j, nz - 1) = 0.5f*(field(nx - 2, j, nz - 1) + field(nx - 1, j, nz - 2));
		}
	}

	__global__ void SetScalarFieldBoundary_xy(Grid1f field, int nx, int ny, int nz)
	{
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (k >= 1 && k < nz - 1)
		{
			field(0, 0, k) = 0.5f*(field(1, 0, k) + field(0, 1, k));
			field(nx - 1, 0, k) = 0.5f*(field(nx - 2, 0, k) + field(nx - 1, 1, k));
			field(0, ny - 1, k) = 0.5f*(field(1, ny - 1, k) + field(0, ny - 2, k));
			field(nx - 1, ny - 1, k) = 0.5f*(field(nx - 2, ny - 1, k) + field(nx - 1, ny - 2, k));
		}
	}

	//__global__ void P_SetScalarFieldBoundary_x(Grid1f field, float s, int nx, int ny, int nz)
	//{

	//	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//	int k = blockIdx.z * blockDim.z + threadIdx.z;
	//	if (j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	//	{

	//		field(0, j, k) = s * field(1, j, k);
	//		field(nx - 1, j, k) = s * field(nx - 2, j, k);
	//	}
	//}

	//__global__ void P_SetScalarFieldBoundary_y(Grid1f field, float s, int nx, int ny, int nz)
	//{

	//	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//	int k = blockIdx.z * blockDim.z + threadIdx.z;

	//	if (i >= 1 && i < nx - 1 && k >= 1 && k < nz - 1)
	//	{

	//		field(i, 0, k) = s * field(i, 1, k);
	//		field(i, ny - 1, k) = s * field(i, ny - 2, k);
	//	}
	//}

	//__global__ void P_SetScalarFieldBoundary_z(Grid1f field, float s, int nx, int ny, int nz)
	//{

	//	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
	//	{

	//		field(i, j, 0) = s * field(i, j, 1);
	//		field(i, j, nz - 1) = s * field(i, j, nz - 2);
	//	}
	//}


	//__global__ void P_SetScalarFieldBoundary_yz(Grid1f field, int nx, int ny, int nz)
	//{
	//	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//	if (i >= 0 && i < nx)
	//	{

	//		field(i, 0, 0) = 0.5f*(field(i, 1, 0) + field(i, 0, 1));
	//		field(i, ny - 1, 0) = 0.5f*(field(i, ny - 2, 0) + field(i, ny - 1, 1));
	//		field(i, 0, nz - 1) = 0.5f*(field(i, 1, nz - 1) + field(i, 0, nz - 2));
	//		field(i, ny - 1, nz - 1) = 0.5f*(field(i, ny - 1, nz - 2) + field(i, ny - 2, nz - 1));
	//	}
	//}

	//__global__ void P_SetScalarFieldBoundary_xz(Grid1f field, int nx, int ny, int nz)
	//{
	//	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//	if (j >= 1 && j < ny - 1)
	//	{

	//		field(0, j, 0) = 0.5f*(field(1, j, 0) + field(0, j, 1));
	//		field(0, j, nz - 1) = 0.5f*(field(1, j, nz - 1) + field(0, j, nz - 2));
	//		field(nx - 1, j, 0) = 0.5f*(field(nx - 2, j, 0) + field(nx - 2, j, 1));
	//		field(nx - 1, j, nz - 1) = 0.5f*(field(nx - 2, j, nz - 1) + field(nx - 1, j, nz - 2));
	//	}
	//}

	//__global__ void P_SetScalarFieldBoundary_xy(Grid1f field, int nx, int ny, int nz)
	//{

	//	int k = blockIdx.z * blockDim.z + threadIdx.z;

	//	if (k >= 1 && k < nz - 1)
	//	{
	//		field(0, 0, k) = 0.5f*(field(1, 0, k) + field(0, 1, k));
	//		field(nx - 1, 0, k) = 0.5f*(field(nx - 2, 0, k) + field(nx - 1, 1, k));
	//		field(0, ny - 1, k) = 0.5f*(field(1, ny - 1, k) + field(0, ny - 2, k));
	//		field(nx - 1, ny - 1, k) = 0.5f*(field(nx - 2, ny - 1, k) + field(nx - 1, ny - 2, k));
	//	}
	//}

	__global__ void P_SetScalarFieldBoundary_xyz(Grid1f field, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (i == 0 && j == 0 && k == 0)
		{
			field(0, 0, 0) = (field(1, 0, 0) + field(0, 1, 0) + field(0, 0, 1)) / 3.0f;

		}
		if (i == 0 && j == 0 && k == nz - 1)
		{
			field(0, 0, nz - 1) = (field(1, 0, nz - 1) + field(0, 1, nz - 1) + field(0, 0, nz - 2)) / 3.0f;
		}
		if (i == 0 && j == ny - 1 && k == 0)
		{
			field(0, ny - 1, 0) = (field(1, ny - 1, 0) + field(0, ny - 2, 0) + field(0, ny - 1, 1)) / 3.0f;
		}
		if (i == nx - 1 && j == 0 && k == nz - 1)
		{
			field(nx - 1, 0, 0) = (field(nx - 2, 0, 0) + field(nx - 1, 1, 0) + field(nx - 1, 0, 1)) / 3.0f;
		}
		if (i == 0 && j == ny - 1 && k == nz - 1)
		{
			field(0, ny - 1, nz - 1) = (field(1, ny - 1, nz - 1) + field(0, ny - 2, nz - 1) + field(0, ny - 1, nz - 2)) / 3.0f;
		}
		if (i == nx - 1 && j == 0 && k == nz - 1)
		{
			field(nx - 1, 0, nz - 1) = (field(nx - 2, 0, nz - 1) + field(nx - 1, 1, nz - 1) + field(nx - 1, 0, nz - 2)) / 3.0f;
		}
		if (i == nx - 1 && j == ny - 1 && k == 0)
		{
			field(nx - 1, ny - 1, 0) = (field(nx - 2, ny - 1, 0) + field(nx - 1, ny - 2, 0) + field(nx - 1, ny - 1, 1)) / 3.0f;
		}
		if (i == nx - 1 && j == ny - 1 && k == nz - 1)
		{
			field(nx - 1, ny - 1, nz - 1) = (field(nx - 2, ny - 1, nz - 1) + field(nx - 1, ny - 2, nz - 1) + field(nx - 1, ny - 1, nz - 2)) / 3.0f;
		}
	}
	/*
	2019/11/14
	author@wdy
	describe:Setting field boundary
	*/
	void PhaseField::PF_setScalarFieldBoundary(bool postive)
	{
		//return;
		float s = postive ? 1.0f : -1.0f;
		//computer
		//x=0
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid_x((ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		SetScalarFieldBoundary_x << <dimGrid_x, dimBlock >> > (m_cuda_phasefield, s, nx, ny, nz);
		cuSynchronize();
		//y=0
		dim3 dimGrid_y((nx + dimBlock.x - 1) / dimBlock.x, (nz + dimBlock.z - 1) / dimBlock.z);
		SetScalarFieldBoundary_y << <dimGrid_y, dimBlock >> > (m_cuda_phasefield, s, nx, ny, nz);
		cuSynchronize();
		//z=0
		dim3 dimGrid_z((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);
		SetScalarFieldBoundary_z << <dimGrid_z, dimBlock >> > (m_cuda_phasefield, s, nx, ny, nz);
		cuSynchronize();
		//xz=0
		dim3 dimGrid_xz((nx + dimBlock.x - 1) / dimBlock.x);
		SetScalarFieldBoundary_xz << <dimGrid_xz, dimBlock >> > (m_cuda_phasefield, nx, ny, nz);
		cuSynchronize();
		//yz=0
		dim3 dimGrid_yz((ny + dimBlock.y - 1) / dimBlock.y);
		SetScalarFieldBoundary_yz << <dimGrid_yz, dimBlock >> > (m_cuda_phasefield, nx, ny, nz);
		cuSynchronize();
		//xy=0
		dim3 dimGrid_xy((nz + dimBlock.z - 1) / dimBlock.z);
		SetScalarFieldBoundary_xy << <dimGrid_xy, dimBlock >> > (m_cuda_phasefield, nx, ny, nz);
		cuSynchronize();
		//dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		//P_SetScalarFieldBoundary_xyz << <dimGrid, dimBlock >> > (m_cuda_phasefield0, nx, ny, nz);
		//cuSynchronize();

	}







	/*******************************************************************************/
	/*************************************N-S_Solver*********************************/
	/*******************************************************************************/
	/*
	2019/10/27
	author@wdy
	describe: Apply gravity
	*/
	__global__ void P_ApplyGravityForce(Grid1f Velu_x, Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;


		float gg = -9.81f;

		if (i >= 1 && i < nx - 1 && j >= 2 && j < ny - 2 && k >= 1 && k < nz - 1)
		{
			Velu_x(i, j, k) += 0.0f;
			Velv_y(i, j, k) += gg*dt;
			Velw_z(i, j, k) += 0.0f;

		}

	}


	/*
	2019/10/27
	author@wdy
	describe: Advection velecity
	*/
	__global__ void P_InterpolateVelocity(Grid3f Velu_c, Grid1f Velu_u, Grid1f Velv_v, Grid1f Velw_w, int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		int index;
		float3 vel_ijk;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			vel_ijk.x = 0.5f*(Velu_u(i, j, k) + Velu_u(i + 1, j, k));
			vel_ijk.y = 0.5f*(Velv_v(i, j, k) + Velv_v(i, j + 1, k));
			vel_ijk.z = 0.5f*(Velw_w(i, j, k) + Velw_w(i, j, k + 1));

			Velu_c(i, j, k).x = vel_ijk.x;
			Velu_c(i, j, k).y = vel_ijk.y;
			Velu_c(i, j, k).z = vel_ijk.z;
		}
	}

	__global__ void P_AdvectionVelocity(Grid3f vel_k, Grid3f vel_k0, int nx, int ny, int nz, float dt)
	{
		float h = 0.005f;
		float fx, fy, fz;
		int  ix, iy, iz;
		float w000, w100, w010, w001, w111, w011, w101, w110;
		int index;
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{

			fx = i + dt*vel_k0(i, j, k).x / h;
			fy = j + dt*vel_k0(i, j, k).y / h;
			fz = k + dt*vel_k0(i, j, k).z / h;

			if (fx < 1) { fx = 1; }
			if (fx > nx - 2) { fx = nx - 2; }
			if (fy < 1) { fy = 1; }
			if (fy > ny - 2) { fy = ny - 2; }
			if (fz < 1) { fz = 1; }
			if (fz > nz - 2) { fz = nz - 2; }

			ix = (int)fx;
			iy = (int)fy;
			iz = (int)fz;
			fx -= ix;
			fy -= iy;
			fz -= iz;

			//float& val = d0(i,j,k);
			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			w100 = fx * (1.0f - fy)*(1.0f - fz);
			w010 = (1.0f - fx)*fy *(1.0f - fz);
			w001 = (1.0f - fx)*(1.0f - fy)*fz;
			w111 = fx * fy * fz;
			w011 = (1.0f - fx)*fy * fz;
			w101 = fx * (1.0f - fy)*fz;
			w110 = fx * fy *(1.0f - fz);
			//原子操作

			//x direction
			atomicAdd(&vel_k(ix, iy, iz).x, vel_k0(i, j, k).x * w000);
			atomicAdd(&vel_k(ix + 1, iy, iz).x, vel_k0(i, j, k).x * w100);
			atomicAdd(&vel_k(ix, iy + 1, iz).x, vel_k0(i, j, k).x * w010);
			atomicAdd(&vel_k(ix, iy, iz + 1).x, vel_k0(i, j, k).x * w001);

			atomicAdd(&vel_k(ix + 1, iy + 1, iz + 1).x, vel_k0(i, j, k).x * w111);
			atomicAdd(&vel_k(ix, iy + 1, iz + 1).x, vel_k0(i, j, k).x * w011);
			atomicAdd(&vel_k(ix + 1, iy, iz + 1).x, vel_k0(i, j, k).x * w101);
			atomicAdd(&vel_k(ix + 1, iy + 1, iz).x, vel_k0(i, j, k).x * w110);

			//y direction
			atomicAdd(&vel_k(ix, iy, iz).y, vel_k0(i, j, k).y * w000);
			atomicAdd(&vel_k(ix + 1, iy, iz).y, vel_k0(i, j, k).y * w100);
			atomicAdd(&vel_k(ix, iy + 1, iz).y, vel_k0(i, j, k).y * w010);
			atomicAdd(&vel_k(ix, iy, iz + 1).y, vel_k0(i, j, k).y * w001);

			atomicAdd(&vel_k(ix + 1, iy + 1, iz + 1).y, vel_k0(i, j, k).y * w111);
			atomicAdd(&vel_k(ix, iy + 1, iz + 1).y, vel_k0(i, j, k).y * w011);
			atomicAdd(&vel_k(ix + 1, iy, iz + 1).y, vel_k0(i, j, k).y * w101);
			atomicAdd(&vel_k(ix + 1, iy + 1, iz).y, vel_k0(i, j, k).y * w110);

			//z direction
			atomicAdd(&vel_k(ix, iy, iz).z, vel_k0(i, j, k).z * w000);
			atomicAdd(&vel_k(ix + 1, iy, iz).z, vel_k0(i, j, k).z * w100);
			atomicAdd(&vel_k(ix, iy + 1, iz).z, vel_k0(i, j, k).z * w010);
			atomicAdd(&vel_k(ix, iy, iz + 1).z, vel_k0(i, j, k).z * w001);

			atomicAdd(&vel_k(ix + 1, iy + 1, iz + 1).z, vel_k0(i, j, k).z * w111);
			atomicAdd(&vel_k(ix, iy + 1, iz + 1).z, vel_k0(i, j, k).z * w011);
			atomicAdd(&vel_k(ix + 1, iy, iz + 1).z, vel_k0(i, j, k).z * w101);
			atomicAdd(&vel_k(ix + 1, iy + 1, iz).z, vel_k0(i, j, k).z * w110);

			//fx = i + dt*Velu_c0(i, j, k).x / h;
			//fy = j + dt*Velu_c0(i, j, k).y / h;
			//fz = k + dt*Velu_c0(i, j, k).z / h;

			//if (fx < 1) { fx = 1; }
			//if (fx > nx - 2) { fx = nx - 2; }
			//if (fy < 1) { fy = 1; }
			//if (fy > ny - 2) { fy = ny - 2; }
			//if (fz < 1) { fz = 1; }
			//if (fz > nz - 2) { fz = nz - 2; }

			//ix = (int)fx;
			//iy = (int)fy;
			//iz = (int)fz;
			//fx -= ix;
			//fy -= iy;
			//fz -= iz;

			////float& val = d0(i,j,k);
			//w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			//w100 = fx * (1.0f - fy)*(1.0f - fz);
			//w010 = (1.0f - fx)*fy *(1.0f - fz);
			//w001 = (1.0f - fx)*(1.0f - fy)*fz;
			//w111 = fx * fy * fz;
			//w011 = (1.0f - fx)*fy * fz;
			//w101 = fx * (1.0f - fy)*fz;
			//w110 = fx * fy *(1.0f - fz);
			////原子操作

			////x direction
			//atomicAdd(&Velu_c(ix, iy, iz).x, Velu_c0(i, j, k).x * w000);
			//atomicAdd(&Velu_c(ix + 1, iy, iz).x, Velu_c0(i, j, k).x * w100);
			//atomicAdd(&Velu_c(ix, iy + 1, iz).x, Velu_c0(i, j, k).x * w010);
			//atomicAdd(&Velu_c(ix, iy, iz + 1).x, Velu_c0(i, j, k).x * w001);

			//atomicAdd(&Velu_c(ix + 1, iy + 1, iz + 1).x, Velu_c0(i, j, k).x * w111);
			//atomicAdd(&Velu_c(ix, iy + 1, iz + 1).x, Velu_c0(i, j, k).x * w011);
			//atomicAdd(&Velu_c(ix + 1, iy, iz + 1).x, Velu_c0(i, j, k).x * w101);
			//atomicAdd(&Velu_c(ix + 1, iy + 1, iz).x, Velu_c0(i, j, k).x * w110);

			////y direction
			//atomicAdd(&Velu_c(ix, iy, iz).y, Velu_c0(i, j, k).y * w000);
			//atomicAdd(&Velu_c(ix + 1, iy, iz).y, Velu_c0(i, j, k).y * w100);
			//atomicAdd(&Velu_c(ix, iy + 1, iz).y, Velu_c0(i, j, k).y * w010);
			//atomicAdd(&Velu_c(ix, iy, iz + 1).y, Velu_c0(i, j, k).y * w001);

			//atomicAdd(&Velu_c(ix + 1, iy + 1, iz + 1).y, Velu_c0(i, j, k).y * w111);
			//atomicAdd(&Velu_c(ix, iy + 1, iz + 1).y, Velu_c0(i, j, k).y * w011);
			//atomicAdd(&Velu_c(ix + 1, iy, iz + 1).y, Velu_c0(i, j, k).y * w101);
			//atomicAdd(&Velu_c(ix + 1, iy + 1, iz).y, Velu_c0(i, j, k).y * w110);

			////z direction
			//atomicAdd(&Velu_c(ix, iy, iz).z, Velu_c0(i, j, k).z * w000);
			//atomicAdd(&Velu_c(ix + 1, iy, iz).z, Velu_c0(i, j, k).z * w100);
			//atomicAdd(&Velu_c(ix, iy + 1, iz).z, Velu_c0(i, j, k).z * w010);
			//atomicAdd(&Velu_c(ix, iy, iz + 1).z, Velu_c0(i, j, k).z * w001);

			//atomicAdd(&Velu_c(ix + 1, iy + 1, iz + 1).z, Velu_c0(i, j, k).z * w111);
			//atomicAdd(&Velu_c(ix, iy + 1, iz + 1).z, Velu_c0(i, j, k).z * w011);
			//atomicAdd(&Velu_c(ix + 1, iy, iz + 1).z, Velu_c0(i, j, k).z * w101);
			//atomicAdd(&Velu_c(ix + 1, iy + 1, iz).z, Velu_c0(i, j, k).z * w110);

		}
	}

	__global__ void P_InterpolatedVelocity(Grid3f Velu_c, Grid1f Velu_x, Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		float vx, vy, vz;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			vx = 0.5f*(Velu_c(i, j, k).x + Velu_c(i + 1, j, k).x);
			vy = 0.5f*(Velu_c(i, j, k).y + Velu_c(i, j + 1, k).y);
			vz = 0.5f*(Velu_c(i, j, k).z + Velu_c(i, j, k + 1).z);

			Velu_x(i, j, k) = vx;
			Velv_y(i, j, k) = vy;
			Velw_z(i, j, k) = vz;
		}
	}



	/*
	2019/10/27
	author@wdy
	describe: Set boundary
	*/
	__global__ void P_SetU(Grid1f Velu_x, int nx, int ny, int nz)
	{
		/*	int i = blockDim.x * blockIdx.x + threadIdx.x;*/
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (j >= 0 && j < ny && k >= 0 && k < nz)
		{
			Velu_x(0, j, k) = 0.0f;
			Velu_x(1, j, k) = 0.0f;
			Velu_x(nx, j, k) = 0.0f;
			Velu_x(nx - 1, j, k) = 0.0f;
		}
	}

	__global__ void P_SetV(Grid1f Velv_y, int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int k = blockIdx.z * blockDim.z + threadIdx.z;


		if (i >= 0 && i < nx && k >= 0 && k < nz)
		{
			Velv_y(i, 0, k) = 0.0f;
			Velv_y(i, 1, k) = 0.0f;
			Velv_y(i, ny, k) = 0.0f;
			Velv_y(i, ny - 1, k) = 0.0f;
		}
	}


	__global__ void P_SetW(Grid1f Velw_z, int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		//int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= 0 && i < nx && j >= 0 && j < ny)
		{
			Velw_z(i, j, 0) = 0.0f;
			Velw_z(i, j, 1) = 0.0f;
			Velw_z(i, j, nz) = 0.0f;
			Velw_z(i, j, nz - 1) = 0.0f;
		}
	}

	/*
	2019/10/27
	author@wdy
	describe: Solve divergence and  coefficient
	*/
	__global__ void P_PrepareForProjection(GridCoef coefMatrix, Grid1f RHS, Grid1f mass, Grid1f Velu_x, Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		//if (i >= nx) return;
		//if (j >= ny) return;
		//if (k >= nz) return;

		//float h = pfParams.h;
		float h = 0.005f;
		float hh = h*h;
		float div_ijk = 0.0f;
		float S = 0.2f;
		Coef A_ijk;
		//int index;
		A_ijk.a = 0.0f;
		A_ijk.x0 = 0.0f;
		A_ijk.x1 = 0.0f;
		A_ijk.y0 = 0.0f;
		A_ijk.y1 = 0.0f;
		A_ijk.z0 = 0.0f;
		A_ijk.z1 = 0.0f;


		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			float m_ijk = mass(i, j, k);
			if (i < nx - 2) {
				float c = 0.5f*(m_ijk + mass(i + 1, j, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));//分母是密度，c是phi

				A_ijk.a += term;
				A_ijk.x1 += term;
			}
			div_ijk -= Velu_x(i + 1, j, k) / h;
			//left neighbour
			if (i > 1) {
				float c = 0.5f*(m_ijk + mass(i - 1, j, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.x0 += term;
			}

			div_ijk += Velu_x(i, j, k) / h;

			//top neighbour
			if (j < ny - 2) {
				float c = 0.5f*(m_ijk + mass(i, j + 1, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.y1 += term;
			}

			div_ijk -= Velv_y(i, j + 1, k) / h;
			//bottom neighbour
			if (j > 1) {
				float c = 0.5f*(m_ijk + mass(i, j - 1, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.y0 += term;
			}

			div_ijk += Velv_y(i, j, k) / h;
			//far neighbour

			if (k < nz - 2) {
				float c = 0.5f*(m_ijk + mass(i, j, k + 1));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));
				A_ijk.a += term;
				A_ijk.z1 += term;

			}
			div_ijk -= Velw_z(i, j, k + 1) / h;

			//near neighbour
			if (k > 1) {
				float c = 0.5f*(m_ijk + mass(i, j, k - 1));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.z0 += term;
			}

			div_ijk += Velw_z(i, j, k) / h;
			if (m_ijk > 1.0)
			{
				div_ijk += 0.5f*pow((m_ijk - 1.0f), 1.0f) / dt;
				div_ijk += S*((mass(i + 1, j, k) - m_ijk)+ (mass(i - 1, j, k) - m_ijk)+ (mass(i, j + 1, k) - m_ijk)+ (mass(i, j - 1, k) - m_ijk) + (mass(i, j, k + 1) - m_ijk)+  (mass(i, j, k - 1) - m_ijk)) / m_ijk / dt;
			}

			coefMatrix(i, j, k) = A_ijk;
			RHS(i, j, k) = div_ijk;//度散
		}

		//	float m_ijk = mass(i, j, k);
		//	if (i < nx - 2) {
		//		float c = 0.5f*(m_ijk + mass(i + 1, j, k));
		//		c = c > 1.0f ? 1.0f : c;
		//		c = c < 0.0f ? 0.0f : c;
		//		float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));//分母是密度，c是phi

		//		A_ijk.a += term;
		//		A_ijk.x1 += term;
		//	}

		//	div_ijk -= Velu_x(i + 1, j, k) / h;
		//	if (i > 1) {
		//		float c = 0.5f*(m_ijk + mass(i - 1, j, k));
		//		c = c > 1.0f ? 1.0f : c;
		//		c = c < 0.0f ? 0.0f : c;
		//		float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

		//		A_ijk.a += term;
		//		A_ijk.x0 += term;
		//	}
		//	div_ijk += Velu_x(i, j, k) / h;




		//	if (j < ny - 2) {
		//		float c = 0.5f*(m_ijk + mass(i, j + 1, k));
		//		c = c > 1.0f ? 1.0f : c;
		//		c = c < 0.0f ? 0.0f : c;
		//		float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

		//		A_ijk.a += term;
		//		A_ijk.y1 += term;
		//	}
		//	div_ijk -= Velv_y(i, j + 1, k) / h;

		//	if (j > 1) {
		//		float c = 0.5f*(m_ijk + mass(i, j - 1, k));
		//		c = c > 1.0f ? 1.0f : c;
		//		c = c < 0.0f ? 0.0f : c;
		//		float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

		//		A_ijk.a += term;
		//		A_ijk.y0 += term;
		//	}

		//	div_ijk += Velv_y(i, j, k) / h;


		//	if (k < nz - 2) {
		//		float c = 0.5f*(m_ijk + mass(i, j, k + 1));
		//		c = c > 1.0f ? 1.0f : c;
		//		c = c < 0.0f ? 0.0f : c;
		//		float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));
		//		A_ijk.a += term;
		//		A_ijk.z1 += term;

		//	}
		//	div_ijk -= Velw_z(i, j, k + 1) / h;

		//	if (k > 1) {
		//		float c = 0.5f*(m_ijk + mass(i, j, k - 1));
		//		c = c > 1.0f ? 1.0f : c;
		//		c = c < 0.0f ? 0.0f : c;
		//		float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

		//		A_ijk.a += term;
		//		A_ijk.z0 += term;
		//	}
		//	div_ijk += Velw_z(i, j, k) / h;



		//	if (m_ijk > 1.0)
		//	{
		//		div_ijk += 0.5f*pow((m_ijk - 1.0f), 1.0f) / dt;
		//		//div_ijk += S*((mass(i + 1, j, k) - m_ijk)+ (mass(i - 1, j, k) - m_ijk)+ (mass(i, j + 1, k) - m_ijk)+ (mass(i, j - 1, k) - m_ijk) + (mass(i, j, k + 1) - m_ijk)+  (mass(i, j, k - 1) - m_ijk)) / m_ijk / dt;
		//	}

		//	coefMatrix(i, j, k) = A_ijk;
		//	RHS(i, j, k) = div_ijk;//度散

		//}
	}

	/*
	2019/10/27
	author@wdy
	describe:Solve pressure
	*/
	__global__ void P_Projection(Grid1f pressure, Grid1f bufPressure, GridCoef coefMatrix, Grid1f RHS, int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			Coef A_ijk = coefMatrix(i, j, k);

			float a = A_ijk.a;
			float x0 = A_ijk.x0;
			float x1 = A_ijk.x1;
			float y0 = A_ijk.y0;
			float y1 = A_ijk.y1;
			float z0 = A_ijk.z0;
			float z1 = A_ijk.z1;
			float p_ijk;

			p_ijk = RHS(i, j, k);
			if (i > 0) p_ijk += x0*bufPressure(i - 1, j, k);
			if (i < nx - 1) p_ijk += x1*bufPressure(i + 1, j, k);
			if (j > 0) p_ijk += y0*bufPressure(i, j - 1, k);
			if (j < ny - 1) p_ijk += y1*bufPressure(i, j + 1, k);
			if (k > 0) p_ijk += z0*bufPressure(i, j, k - 1);
			if (k < nz - 1) p_ijk += z1*bufPressure(i, j, k + 1);

			pressure(i, j, k) = p_ijk / a;
		}

	}


	__global__ void P_UpdateVelocity_U(Grid1f Velu_x, Grid1f pressure, Grid1f mass, int nx, int ny, int nz, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;


		int ni = nx;
		int nj = ny;
		int nk = nz;
		int nij = ni*nj;
		float h = 0.005f;
		//int index;
		if (i >= 2 && i < Velu_x.nx - 2 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{

			float c = 0.5f*(mass(i - 1, j, k) + mass(i, j, k));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;

			Velu_x(i, j, k) -= dt*(pressure(i, j, k) - pressure(i - 1, j, k)) / h / (c*RHO1 + (1.0f - c)*RHO2);


		}


		//if (i >= 2 && i < (nx + 1) - 2 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		//{


		//	float c = 0.5f*(mass(i - 1, j, k) + mass(i, j, k));
		//	c = c > 1.0f ? 1.0f : c;
		//	c = c < 0.0f ? 0.0f : c;

		//	Velu_x(i, j, k) -= dt*(pressure(i, j, k) - pressure(i - 1, j, k)) / h / (c*RHO1 + (1.0f - c)*RHO2);


		//}
	}

	__global__ void P_UpdateVelocity_V(Grid1f Velv_y, Grid1f pressure, Grid1f mass, int nx, int ny, int nz, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int ni = nx;
		int nj = ny;
		int nk = nz;
		int nij = ni*nj;
		float h = 0.005f;
		int index;
		if (i >= 1 && i < nx - 1 && j >= 2 && j < Velv_y.ny - 2 && k >= 1 && k < nz - 1)
		{



			float c = 0.5f*(mass(i, j, k) + mass(i, j - 1, k));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			Velv_y(i, j, k) -= dt*(pressure(i, j, k) - pressure(i, j - 1, k)) / h / (c*RHO1 + (1.0f - c)*RHO2);
		}
		//if (i >= 1 && i < nx - 1 && j >= 2 && j < (ny + 1) - 2 && k >= 1 && k < nz - 1)
		//{

		//	float c = 0.5f*(mass(i, j, k) + mass(i, j - 1, k));
		//	c = c > 1.0f ? 1.0f : c;
		//	c = c < 0.0f ? 0.0f : c;
		//	Velv_y(i, j, k) -= dt*(pressure(i, j, k) - pressure(i, j - 1, k)) / h / (c*RHO1 + (1.0f - c)*RHO2);

		//}



	}

	__global__ void P_UpdateVelocity_W(Grid1f Velw_z, Grid1f pressure, Grid1f mass, int nx, int ny, int nz, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;


		int ni = nx;
		int nj = ny;
		int nk = nz;
		int nij = ni*nj;
		float h = 0.005f;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 2 && k < Velw_z.nz - 2)
		{
			float c = 0.5f*(mass(i, j, k) + mass(i, j, k - 1));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			Velw_z(i, j, k) -= dt*(pressure(i, j, k) - pressure(i, j, k - 1)) / h / (c*RHO1 + (1.0f - c)*RHO2);
		}

		//if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 2 && k < (nz + 1) - 2)
		//{


		//	float c = 0.5f*(mass(i, j, k) + mass(i, j, k - 1));
		//	c = c > 1.0f ? 1.0f : c;
		//	c = c < 0.0f ? 0.0f : c;
		//	Velw_z(i, j, k) -= dt*(pressure(i, j, k) - pressure(i, j, k - 1)) / h / (c*RHO1 + (1.0f - c)*RHO2);


		//}

	}
	/*
	2019/11/13
	author@wdy
	describe:N-S equation solver
	*/
	void PhaseField::NS_solver(float substep)
	{
		//cout << m_cuda_Velu.nx << endl;
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		P_ApplyGravityForce << < dimGrid, dimBlock >> > (m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, substep);
		cuSynchronize();

		//Interpolation from Boundary to Center
		P_InterpolateVelocity << < dimGrid, dimBlock >> > (m_cuda_Veluc0, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz);
		cuSynchronize();
		//Semi-Lagrangian Advection
		P_AdvectionVelocity << < dimGrid, dimBlock >> > (m_cuda_Veluc, m_cuda_Veluc0, nx, ny, nz, substep);
		cuSynchronize();
		//Interpolation from Center to Boundary
		P_InterpolatedVelocity << < dimGrid, dimBlock >> > (m_cuda_Veluc, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, substep);
		cuSynchronize();

		dim3 dimGrid_x((ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		P_SetU << < dimGrid_x, dimBlock >> > (m_cuda_Velu, nx, ny, nz);
		dim3 dimGrid_y((nx + dimBlock.x - 1) / dimBlock.x, (nz + dimBlock.z - 1) / dimBlock.z);
		P_SetV << < dimGrid_y, dimBlock >> > (m_cuda_Velv, nx, ny, nz);
		dim3 dimGrid_z((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);
		P_SetW << < dimGrid_z, dimBlock >> > (m_cuda_Velw, nx, ny, nz);

		P_PrepareForProjection << < dimGrid, dimBlock >> > (m_cuda_CoefMatrix, m_cuda_RHS, m_cuda_phasefield0, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, substep);
		cuSynchronize();
		m_cuda_Pressure.cudaClear();
		//雅克比迭代求解压力
		for (int i = 0; i < 1800; i++)
		{
			K_CopyGData << < dimGrid, dimBlock >> > (m_cuda_BufPressure, m_cuda_Pressure, nx, ny, nz);
			P_Projection << < dimGrid, dimBlock >> > (m_cuda_Pressure, m_cuda_BufPressure, m_cuda_CoefMatrix, m_cuda_RHS, nx, ny, nz);
			cuSynchronize();
		}

		P_UpdateVelocity_U << < dimGrid, dimBlock >> > (m_cuda_Velu, m_cuda_Pressure, m_cuda_phasefield0, nx, ny, nz, substep);
		cuSynchronize();
		P_UpdateVelocity_V << < dimGrid, dimBlock >> > (m_cuda_Velv, m_cuda_Pressure, m_cuda_phasefield0, nx, ny, nz, substep);
		cuSynchronize();
		P_UpdateVelocity_W << < dimGrid, dimBlock >> > (m_cuda_Velw, m_cuda_Pressure, m_cuda_phasefield0, nx, ny, nz, substep);
		cuSynchronize();
	}



	void PhaseField::display()
	{
		glEnableClientState(GL_INDEX_ARRAY);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);


		//glBindBuffer(GL_ARRAY_BUFFER, SimulationRegionColor_bufferObj);
		//glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(rgb), 0);

		//glBindBuffer(GL_ARRAY_BUFFER, SimulationRegion_bufferObj);
		//glVertexPointer(3, GL_FLOAT, sizeof(Grid4f), 0);

		//glEnable(GL_BLEND);
		//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		//glDrawElements(GL_POINTS, 4 * (nx - 1)*(ny - 1)*(nz - 1), GL_UNSIGNED_INT, 0);
		//glDisable(GL_BLEND);


		glBindBuffer(GL_ARRAY_BUFFER, Color_bufferObj);
		glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(uchar4), 0);

		glBindBuffer(GL_ARRAY_BUFFER, Initpos_bufferObj);
		glVertexPointer(3, GL_FLOAT, sizeof(float3), 0);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDrawElements(GL_POINTS, 4 * (nx - 1)*(ny - 1)*(nz - 1), GL_UNSIGNED_INT, 0);
		glDisable(GL_BLEND);

		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_INDEX_ARRAY);

	}
}