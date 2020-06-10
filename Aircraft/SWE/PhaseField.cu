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
#define grid2Dwrite(array, x, y, pitch, value) array[(y)*pitch+x] = value
#define grid2Dread(array, x, y, pitch) array[(y)*pitch+x]
	//float m_virtualGridSize = 0.1;
	//float samplingDistance = 0.02f;
	/*
	date:2019/11/14
	author:@wdy
	describe:Data copy on GPU
	*/


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

		AllocateMemoery(50, 60, 50);


	}

	PhaseField::~PhaseField(void)
	{
	}

	void PhaseField::initialize()
	{
		diff = 0.0f;

		initRegion();
	}




	__global__ void C_InitPosition(Grid3f position, rgb* color, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		int index;
		float h = 0.1f;
		if (i >= 0 && i < nx  && j >= 0 && j < ny && k >= 0 && k < nz)
		{

			float x = i * h;
			float y = j * h;
			float z = k * h;

			position(i, j, k).x = x;
			position(i, j, k).y = y;
			position(i, j, k).z = z;


			//printf("(%d,%d,%d) =%f\n", i, j, k, y);
		}
	}


	__global__ void C_InitPhaseField(Grid3f position, Grid1f phasefield0, int nx, int ny, int nz)
	{
		int index;
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			index = i + j*nx + k*nx*ny;
			//if (i ==50 &&j ==50 &&k ==50)
			if (i > 10 && i < 30 && j > 10 && j < 50 && k>10 && k < 30)
				//if (d<85.0f)
			{
				phasefield0(i, j, k) = 1.0f;
			}
			else
			{
				phasefield0(i, j, k) = 0.0f;
			}
		}
	}


	__global__ void C_PhaseChange(Grid1f phasefield, rgb* color, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		int index;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			index = i + j*nx + k*nx*ny;
			color[index] = make_uchar4(120, 0, 60, phasefield(i, j, k) * 255);
		}
	}




	void PhaseField::initRegion()
	{

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		C_InitPosition << < dimGrid, dimBlock >> > (m_cuda_position, m_cuda_color, nx, ny, nz);
		synchronCheck;

		C_InitPhaseField << <dimGrid, dimBlock >> > (m_cuda_position, m_cuda_phasefield0, nx, ny, nz);
		synchronCheck;


	}
	//__global__ void C_MoveSimulatedRegion(Grid3f position, Grid1f vel_v, gridpoint* m_mdevice_grid, size_t pitch, float4* displacement,int nx, int ny, int nz)
	//{
	//	int i = threadIdx.x + blockIdx.x * blockDim.x;
	//	int j = threadIdx.y + blockIdx.y * blockDim.y;
	//	int k = threadIdx.z + blockIdx.z * blockDim.z;
	//	float vel=0.0f;
	//	float4 gp;
	//	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	//	{
	//		gp = displacement[i + j*nx + k*nx*ny];
	//		//float4 gp = grid2Dread(m_mdevice_grid, i, k, pitch);
	//		float h = 2.0f;
	//		if (position(i, j, k).y < gp.x)
	//		{


	//		vel =vel+ 0.5*(gp.y + gp.y);
	//		//vel_v(i, j, k) = vel;
	//		gp.z = vel;
	//		grid2Dwrite(m_mdevice_grid, i + 1, k + 1, pitch, gp);

	//		}



	//	}
	//}


	__global__ void C_moveSimulationRegion(Grid3f m_position, int nx, int ny, int nz, int dx, int dy, glm::vec3 v)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int width = nx;
		int height = nz;
		int index;

		if (i >= 0 && i < nx - 1 && j >= 0 && j < ny - 1 && k >= 0 && k < nz - 1)
		{

			float3 gp = m_position(i, j, k);
			float4 gp_init = make_float4(gp.x, gp.y, gp.z, gp.z);
			int new_i = i - dx;
			int new_j = j - dy;


			gp.x = new_i < 0 || new_i >= width ? gp_init.x : gp.x;
			new_i = new_i % width;
			new_i = new_i < 0 ? width + new_i : new_i;

			gp.z = new_j < 0 || new_j >= height ? gp_init.z : gp.z;
			new_j = new_j % height;
			new_j = new_j < 0 ? height + new_j : new_j;

			/*m_position(new_i+1, j, new_j+1).x += v.x*0.0016;
			m_position(new_i+1, j, new_j+1).z += v.z*0.0016;*/

			m_position(new_i, j, new_j).x = gp.x;
			m_position(new_i, j, new_j).z = gp.z;

		}
	}

	void PhaseField::moveDynamicRegion(int dx, int dy, glm::vec3 v)
	{
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);

		C_moveSimulationRegion << < dimGrid, dimBlock >> > (m_cuda_position, nx, ny, nz, dx, dy, v);
		synchronCheck;

		p_simulatedOriginX += dx;
		p_simulatedOriginY += dy;

		//	std::cout << "Origin X: " << m_simulatedOriginX << " Origin Y: " << m_simulatedOriginY << std::endl;
	}

	/*
	date:2019/12/07
	author:@wdy
	describe:allocate memoery for all variable
	*/
	void PhaseField::AllocateMemoery(int _nx, int _ny, int _nz)
	{

		nx = _nx;//计算区域-流体部分
		ny = _ny;
		nz = _nz;

		dSize = nx*ny*nz;
		//for phasefield equation
		m_cuda_position.cudaSetSpace(nx, ny, nz);
		m_cuda_phasefield0.cudaSetSpace(nx, ny, nz);
		m_cuda_phasefield.cudaSetSpace(nx, ny, nz);
		m_cuda_phasefieldcp.cudaSetSpace(nx, ny, nz);

		cudaCheck(cudaMalloc(&m_cuda_SimulationRegionColor, dSize * sizeof(rgb)));
		cudaCheck(cudaMalloc(&m_cuda_color, dSize * sizeof(uchar4)));
		cudaCheck(cudaMalloc(&max_velu, 1 * sizeof(float)));
		cfl = (float*)malloc(1 * sizeof(float));

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
		m_cuda_Norm.cudaSetSpace(nx, ny, nz);

		//粒子位置、相场及颜色
		size_t size;
		glGenBuffers(1, &Initpos_bufferObj);
		glBindBuffer(GL_ARRAY_BUFFER, Initpos_bufferObj);
		glBufferData(GL_ARRAY_BUFFER, dSize * sizeof(float3), NULL, GL_DYNAMIC_COPY);
		cudaGraphicsGLRegisterBuffer(&Initpos_resource, Initpos_bufferObj, cudaGraphicsMapFlagsWriteDiscard);
		cudaGraphicsMapResources(1, &Initpos_resource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&m_cuda_position.data, &size, Initpos_resource);

		glGenBuffers(1, &Color_bufferObj);
		glBindBuffer(GL_ARRAY_BUFFER, Color_bufferObj);
		glBufferData(GL_ARRAY_BUFFER, dSize * sizeof(uchar4), NULL, GL_STATIC_DRAW);
		cudaGraphicsGLRegisterBuffer(&Color_resource, Color_bufferObj, cudaGraphicsMapFlagsWriteDiscard);
		cudaGraphicsMapResources(1, &Color_resource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&m_cuda_color, &size, Color_resource);
	}


	/*******************************************************************************/
	/*************************************N-S_Solver*********************************/
	/*******************************************************************************/

	/*
	2019/10/27
	author@wdy
	describe: Apply gravity
	*/


	__global__ void P_InitailVelecity(Grid1f Velu_x, Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		float gg = -9.81f;

		if (i >= 1 && i < nx - 1 && j >= 0 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			Velu_x(i, j, k) = 0.0;
			Velv_y(i, j, k) = 0.0;
			Velw_z(i, j, k) = 0.0;

		}

	}



	__global__ void P_ApplyGravityForce(Grid1f Velu_x, Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		float gg = -9.81f;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			Velu_x(i, j, k) += 0.0;
			Velv_y(i, j, k) += gg*dt;
			Velw_z(i, j, k) += 0.0;
			////if (abs(Velv_y(i, j, k)) > 13)
			//printf("Y方向速度=%f", Velv_y(i, j, k));
			//printf("(%d,%d,%d) =%f\n", i, j, k, Velv_y(i, j, k));
		}

	}

	__global__ void P_InterpolateVelocity(Grid3f Velu_c0, Grid1f Velu_x, Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		int index;
		float3 vel_ijk;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			index = i + j*nx + k*nx*ny;
			vel_ijk.x = 0.5f*(Velu_x(i, j, k) + Velu_x(i + 1, j, k));
			vel_ijk.y = 0.5f*(Velv_y(i, j, k) + Velv_y(i, j + 1, k));
			vel_ijk.z = 0.5f*(Velw_z(i, j, k) + Velw_z(i, j, k + 1));

			Velu_c0(i, j, k).x = vel_ijk.x;
			Velu_c0(i, j, k).y = vel_ijk.y;
			Velu_c0(i, j, k).z = vel_ijk.z;

			//printf("(%d,%d,%d) =%f\n", i, j, k, Velu_c0(i, j, k).y);
			//if (abs(Velv_v(i, j, k))>100)
			//printf("Y方向速度=%f", Velu_c0(i, j, k).y);
		}
	}

	__global__ void P_AdvectionVelocity(Grid3f vel_k, Grid3f vel_k0, int nx, int ny, int nz, float dt)
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
			fx = i + dt*vel_k0(i, j, k).x / samplingDistance;
			fy = j + dt*vel_k0(i, j, k).y / samplingDistance;
			fz = k + dt*vel_k0(i, j, k).z / samplingDistance;

			//float xx = dt*vel_k0(i, j, k).y / samplingDistance;
			//printf("(%d,%d,%d) =%f\n", i,j, k, fy);


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


			//if (abs(vel_k(ix, iy, iz).y>0.5))
			//printf("(%d,%d,%d) =%f\n", i, j, k, vel_k(i, j, k).y);
			//printf("Y方向速度=%f", vel_k(i, j, k).y);

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


			//printf("(%d,%d,%d) =%f\n", i, j, k, Velv_y(i, j, k));
		}
	}


	/*
	2019/10/27
	author@wdy
	describe: Set boundary...
	*/
	__global__ void P_SetU(Grid3f position, Grid1f Velu_x, gridpoint* m_mdevice_grid, size_t pitch, float4* displacement,int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		//float4 gp;
		int gridx;
		int gridy;

		if (j >= 0 && j < ny &&j >= 0 && j < ny&& k >= 0 && k < nz)
		{
			//gp = displacement[j + k*ny];
			gridx = i+1 ;
			gridy = k +1;
			float4 gp = grid2Dread(m_mdevice_grid, gridx, gridy, pitch);
			//gp = m_mdevice_grid[gridx + gridy*512];
			float h = max(gp.x, 0.0f);
			float uh = gp.y;
			float vh = gp.z;

			float h4 = h * h * h * h;
			float u = sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));
			//float4 gp = grid2Dread(m_mdevice_grid, j, k, pitch);
			if (position(i, j, k).y <= h)
			{

				Velu_x(0, j, k) = 0;
				Velu_x(1, j, k) = 0;
				Velu_x(nx, j, k) =0;
				Velu_x(nx - 1, j, k) =0;
			}
		}
	}

	__global__ void P_SetV(Grid3f position, Grid1f Velv_y, gridpoint* m_mdevice_grid, size_t pitch, int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int gridx;
		int gridy;
		if (i >= 0 && i < nx && j>=0&&j<ny&& k >= 0 && k < nz)
		{
			gridx = i+1;
			gridy = k+1;
			float4 gp = grid2Dread(m_mdevice_grid, gridx, gridy, pitch);
			float h = max(gp.x, 0.0f);
			float uh = gp.y;
			float vh = gp.z;

			float h4 = h * h * h * h;
			float u = sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));

			if (position(i, j, k).y <= h)
			{
				Velv_y(i, 0, k) = 0;
				Velv_y(i, 1, k) = 0;
				Velv_y(i, ny, k) = 0;
				Velv_y(i, ny - 1, k) = 0;
			}
		}
	}


	__global__ void P_SetW(Grid3f position, Grid1f Velw_z, gridpoint* m_mdevice_grid, size_t pitch, float4* displacement, int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		float4 gp;
		int gridx;
		int gridy;
		if (i >= 0 && i < nx && j >= 0 && j < ny&& k >= 0 && k < nz)
		{
			gridx = i+1;
			gridy = k+1;
			float4 gp = grid2Dread(m_mdevice_grid, gridx, gridy, pitch);
			//gp = m_mdevice_grid[gridx + gridy*512];
			float h = max(gp.x, 0.0f);
			float uh = gp.y;
			float vh = gp.z;

			float h4 = h * h * h * h;
			float v = sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));
			
			//float4 gp = grid2Dread(m_mdevice_grid, i, j, pitch);
			if (position(i, j, k).y <= h)
			{
				Velw_z(i, j, 0) = 0;
				Velw_z(i, j, 1) = 0;
				Velw_z(i, j, nz) = 0;
				Velw_z(i, j, nz - 1) = 0;

			}


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

		float hh = samplingDistance*samplingDistance;
		float div_ijk = 0.0f;
		float S = 0.5f;
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
			div_ijk -= Velu_x(i + 1, j, k) / samplingDistance;


			//left neighbour
			if (i > 1) {
				float c = 0.5f*(m_ijk + mass(i - 1, j, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.x0 += term;
			}
			div_ijk += Velu_x(i, j, k) / samplingDistance;

			//top neighbour
			if (j < ny - 2) {
				float c = 0.5f*(m_ijk + mass(i, j + 1, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.y1 += term;
			}
			div_ijk -= Velv_y(i, j + 1, k) / samplingDistance;


			//bottom neighbour
			if (j > 1) {
				float c = 0.5f*(m_ijk + mass(i, j - 1, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.y0 += term;
			}
			div_ijk += Velv_y(i, j, k) / samplingDistance;


			//far neighbour
			if (k < nz - 2) {
				float c = 0.5f*(m_ijk + mass(i, j, k + 1));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));
				A_ijk.a += term;
				A_ijk.z1 += term;

			}
			div_ijk -= Velw_z(i, j, k + 1) / samplingDistance;

			//near neighbour
			if (k > 1) {
				float c = 0.5f*(m_ijk + mass(i, j, k - 1));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.z0 += term;
			}
			div_ijk += Velw_z(i, j, k) / samplingDistance;

			//if (m_ijk > 1.0)
			//{
			//	div_ijk += 0.5f*pow((m_ijk - 1.0f), 1.0f) / dt;
			//	div_ijk += S*((mass(i + 1, j, k) - m_ijk) + (mass(i - 1, j, k) - m_ijk) + (mass(i, j + 1, k) - m_ijk) + (mass(i, j - 1, k) - m_ijk) + (mass(i, j, k + 1) - m_ijk) + (mass(i, j, k - 1) - m_ijk)) / m_ijk / dt;
			//}

			coefMatrix(i, j, k) = A_ijk;
			RHS(i, j, k) = div_ijk;
			//printf("(%d,%d,%d) =%f\n", i, j, k, RHS(i, j, k));
		}
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
			//pressure(i, j, k) = 0.0f;
			p_ijk = RHS(i, j, k);
			if (i > 0) p_ijk += x0*bufPressure(i - 1, j, k);
			if (i < nx - 1) p_ijk += x1*bufPressure(i + 1, j, k);
			if (j > 0) p_ijk += y0*bufPressure(i, j - 1, k);
			if (j < ny - 1) p_ijk += y1*bufPressure(i, j + 1, k);
			if (k > 0) p_ijk += z0*bufPressure(i, j, k - 1);
			if (k < nz - 1) p_ijk += z1*bufPressure(i, j, k + 1);

			pressure(i, j, k) = p_ijk / a;

			//if (pressure(i, j, k)>10)
			//	printf("散度=%f", pressure(i, j, k));

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

		//int index;
		if (i >= 2 && i < Velu_x.nx - 2 && j >= 1 && j < Velu_x.ny - 1 && k >= 1 && k < Velu_x.nz - 1)
		{

			float c = 0.5f*(mass(i - 1, j, k) + mass(i, j, k));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;

			Velu_x(i, j, k) -= dt*(pressure(i, j, k) - pressure(i - 1, j, k)) / samplingDistance / (c*RHO1 + (1.0f - c)*RHO2);


		}

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

		int index;
		if (i >= 1 && i < Velv_y.nx - 1 && j >= 2 && j < Velv_y.ny - 2 && k >= 1 && k < Velv_y.nz - 1)
		{



			float c = 0.5f*(mass(i, j, k) + mass(i, j - 1, k));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			Velv_y(i, j, k) -= dt*(pressure(i, j, k) - pressure(i, j - 1, k)) / samplingDistance / (c*RHO1 + (1.0f - c)*RHO2);
			//if(Velv_y(i, j, k)>100)
			//printf("(%d,%d,%d) =%f\n", i, j, k, Velv_y(i, j, k));
		}
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

		if (i >= 1 && i < Velw_z.nx - 1 && j >= 1 && j < Velw_z.ny - 1 && k >= 2 && k < Velw_z.nz - 2)
		{
			float c = 0.5f*(mass(i, j, k) + mass(i, j, k - 1));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			Velw_z(i, j, k) -= dt*(pressure(i, j, k) - pressure(i, j, k - 1)) / samplingDistance / (c*RHO1 + (1.0f - c)*RHO2);
			//if (pressure(i, j, k)>1)
			//printf("散度=%f", pressure(i, j, k));
		}

	}


	/*******************************************************************************/
	/**********************************PhaseField_Solver****************************/
	/*******************************************************************************/
	/*
	2020/4/6
	author@wdy
	describe: diffusion Phi
	*/
	__global__ void DiffusionPhiNorm(Grid1f d0, Grid3f Norm, int nx, int ny, int nz)
	{

		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;


		float eps = 0.000001f;
		float inv_root2 = 1.0f / sqrt(2.0f);
		float inv_root3 = 1.0f / sqrt(3.0f);
		float h = samplingDistance;


		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			float norm_x;
			float norm_y;
			float norm_z;

			norm_x = d0(i + 1, j, k) - d0(i - 1, j, k);//法向
			norm_y = d0(i, j + 1, k) - d0(i, j - 1, k);
			norm_z = d0(i, j, k + 1) - d0(i, j, k - 1);
			// 				
			float norm = sqrt(norm_x*norm_x + norm_y * norm_y + norm_z * norm_z) + eps;
			norm_x /= norm;//归一化操作，得到的是向量的长度
			norm_y /= norm;
			norm_z /= norm;

			Norm(i, j, k).x = norm_x;
			Norm(i, j, k).y = norm_y;
			Norm(i, j, k).z = norm_z;

		}
	}

	__global__ void DiffusionPhi(Grid1f d0, Grid1f d, Grid3f Norm, float ceo2, int nx, int ny, int nz, float dt)
	{

		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		float weight;

		int ix0, iy0, iz0;
		int ix1, iy1, iz1;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{


			ix0 = i;   iy0 = j; iz0 = k;
			ix1 = i + 1; iy1 = j; iz1 = k;
			weight = 1.0f;

			if (ix1 < nx - 1)
			{

				float3 n1 = Norm(ix0, iy0, iz0);
				float3 n2 = Norm(ix1, iy1, iz1);

				float c0 = d0(ix0, iy0, iz0) * (1.0f - d0(ix0, iy0, iz0))*n1.x;
				float c1 = d0(ix1, iy1, iz1) * (1.0f - d0(ix1, iy1, iz1))*n2.x;// phi(1-phi)*n
				float dc = 0.5f*weight*ceo2*dt*(c1 + c0);

				atomicAdd(&d(ix0, iy0, iz0), -dc);
				atomicAdd(&d(ix1, iy1, iz1), dc);
			}

			ix0 = i; iy0 = j;   iz0 = k;
			ix1 = i; iy1 = j + 1; iz1 = k;

			if (iy1 < ny - 1)
			{
				float3 n1 = Norm(ix0, iy0, iz0);
				float3 n2 = Norm(ix1, iy1, iz1);

				float c0 = d0(ix0, iy0, iz0) * (1.0f - d0(ix0, iy0, iz0))*n1.y;
				float c1 = d0(ix1, iy1, iz1) * (1.0f - d0(ix1, iy1, iz1))*n2.y;
				float dc = 0.5f*weight*ceo2*dt*(c1 + c0);

				atomicAdd(&d(ix0, iy0, iz0), -dc);
				atomicAdd(&d(ix1, iy1, iz1), dc);
			}

			ix0 = i; iy0 = j; iz0 = k;
			ix1 = i; iy1 = j; iz1 = k + 1;

			if (iz1 < nz - 1)
			{
				float3 n1 = Norm(ix0, iy0, iz0);
				float3 n2 = Norm(ix1, iy1, iz1);

				float c0 = d0(ix0, iy0, iz0) * (1.0f - d0(ix0, iy0, iz0))*n1.z;
				float c1 = d0(ix1, iy1, iz1) * (1.0f - d0(ix1, iy1, iz1))*n2.z;
				float dc = 0.5f*weight*ceo2*dt*(c1 + c0);

				atomicAdd(&d(ix0, iy0, iz0), -dc);
				atomicAdd(&d(ix1, iy1, iz1), dc);
			}

		}
	}


	__global__ void LinearSolve(Grid1f d0, Grid1f d, Grid1f cp, int nx, int ny, int nz, float c)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		float c1 = 1.0f / c;
		float c2 = (1.0f - c1) / 6.0f;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			d(i, j, k) = (c1*d0(i, j, k) + c2 * (cp(i + 1, j, k) + cp(i - 1, j, k) + cp(i, j + 1, k) + cp(i, j - 1, k) + cp(i, j, k + 1) + cp(i, j, k + 1)));
		}
	}


	/*******************************************************************************/
	/********************************PhaseField_Solver******************************/
	/*******************************************************************************/

	__global__ void P_AdvectWENO1rd(Grid1f d0, Grid1f d, Grid1f Velu, Grid1f Velv, Grid1f Velw, int nx, int ny, int nz, float dt)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int index;
		int index0;
		int index1;
		//float h = 0.005f;
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
			//if(d(i, j, k)>2)
			////if (i==50&&j==50&&k==50)
			//printf("phi=%f", d(i,j,k));


		}
	}


	/*-----------------------------------------------------------------------------*/
	/*-----------------------------------AdvectForward---------------------------- */
	/*-----------------------------------------------------------------------------*/
	__global__ void Kenel_AdvectForward(Grid1f d0, Grid1f d, Grid1f Velu, Grid1f Velv, Grid1f Velw, int nx, int ny, int nz, float dt)
	{
		//float h = 0.005f;
		float fx, fy, fz;
		int  ix, iy, iz;
		float w000, w100, w010, w001, w111, w011, w101, w110;
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			int idx = i + j*nx + k*nx*ny;
			fx = i + dt*Velu(i, j, k) / samplingDistance;
			fy = j + dt*Velv(i, j, k) / samplingDistance;
			fz = k + dt*Velw(i, j, k) / samplingDistance;
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
			//float& val = d0[idx];
			float& val = d0(i, j, k);
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




	__global__ void P_lDivergence(Grid1f Velu_x, Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;



		float totalDivergence = 0.0f;
		if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
		{
			totalDivergence += abs((Velu_x(i + 1, j, k) - Velu_x(i, j, k)) + (Velv_y(i, j + 1, k) - Velv_y(i, j, k)) + (Velw_z(i, j, k + 1) - Velw_z(i, j, k)));
		}
		if (totalDivergence > 1)
			printf("散度=%f\n", totalDivergence);
		//printf("(%d,%d,%d) =%f\n", i, j, k, totalDivergence);
	}

	__global__ void C_CFL(float* max_velu, Grid1f Velu_x, Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz)
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
	void PhaseField::animate(float dt, gridpoint* m_mdevice_grid, size_t pitch, float4* displacement)
	{

		cout << "时间-------------------" << dt << endl;
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
		P_InitailVelecity << < dimGrid, dimBlock >> > (m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz);
		cuSynchronize();
		while (elipse < dt) {
			C_CFL << <dimGrid, dimBlock >> > (max_velu, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz);
			cuSynchronize();
			//CFL(max_velu);
			cudaMemcpy(cfl, max_velu, 1 * sizeof(float), cudaMemcpyDeviceToHost);
			float substep = cfl[0];
			//float substep = 0.001f;// CFL();
			if (elipse + substep > dt)
			{
				substep = (dt - elipse);
			}
			cout << "*********Substep: " << substep << " *********" << endl;
			t_start = clock();

			//N-S solver
			//NS_solver(substep,t);


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
			P_SetU << < dimGrid_x, dimBlock >> > (m_cuda_position, m_cuda_Velu, m_mdevice_grid, pitch, displacement,nx, ny, nz);
			dim3 dimGrid_y((nx + dimBlock.x - 1) / dimBlock.x, (nz + dimBlock.z - 1) / dimBlock.z);
			P_SetV << < dimGrid_y, dimBlock >> > (m_cuda_position,m_cuda_Velv, m_mdevice_grid, pitch, nx, ny, nz);
			dim3 dimGrid_z((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);
			P_SetW << < dimGrid_z, dimBlock >> > (m_cuda_position, m_cuda_Velw, m_mdevice_grid, pitch, displacement, nx, ny, nz);
			cuSynchronize();
			//m_cuda_RHS.cudaClear();
			//m_cuda_CoefMatrix.cudaClear();
			P_PrepareForProjection << < dimGrid, dimBlock >> > (m_cuda_CoefMatrix, m_cuda_RHS, m_cuda_phasefield0, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, dt);
			cuSynchronize();
			m_cuda_Pressure.cudaClear();
			//雅克比迭代求解压力
			for (int i = 0; i < 200; i++)
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

			//P_lDivergence << < dimGrid, dimBlock >> > (m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, dt);
			//cuSynchronize();
			////t_start = clock();

			////m_cuda_phasefield0 = m_cuda_phasefield;
		/*	K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield, m_cuda_phasefield0, nx, ny, nz);*/




			////PF solver
			//PF_solver(substep); 
			float h = samplingDistance;

			float w = 1.0f*h;//fro diffuse parameter
			float gamma = 1.0f;
			float ceo1 = 1.0f*gamma / h;		//for smoothing
			float ceo2 = 1.5f*gamma*w / h / h;	//for sharping
			float dif2 = (ceo1 + diff / h / h)*substep;
			//float a = (ceo1 + dif2 / h / h)*dt;
			float c = 1.0f + 6.0f*dif2;


			K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield, m_cuda_phasefield0, nx, ny, nz);
			PF_setScalarFieldBoundary(m_cuda_phasefield, true);

			//Kenel_AdvectForward << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, substep);
			//adection
			P_AdvectWENO1rd << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, substep);
			cuSynchronize();

			K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, nx, ny, nz);
			synchronCheck;
			////PF_setScalarFieldBoundary(m_cuda_phasefield0,true);

			//DiffusionPhiNorm << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_Norm, nx, ny, nz);
			//cuSynchronize();

			//DiffusionPhi << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_Norm, ceo2,nx, ny, nz, substep);
			//cuSynchronize();
			//K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, nx, ny, nz);
			//synchronCheck;

			//K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield, m_cuda_phasefield0, nx, ny, nz);
			//for (int it = 0; it < 20; it++)
			//{
			//	PF_setScalarFieldBoundary(m_cuda_phasefield, true);
			//	K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefieldcp, m_cuda_phasefield, nx, ny, nz);
			//	LinearSolve << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_phasefieldcp,nx, ny, nz,c);
			//	cuSynchronize();
			//}

			//moveDynamicRegion(m_cuda_Velv, m_mdevice_grid, pitch, displacement);
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

	__global__ void SetScalarFieldBoundary_yz(Grid1f field, int nx, int ny, int nz)
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
	void PhaseField::PF_setScalarFieldBoundary(Grid1f phasefield, bool postive)
	{
		//return;
		float s = postive ? 1.0f : -1.0f;
		//computer
		//x=0
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid_x((ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		SetScalarFieldBoundary_x << <dimGrid_x, dimBlock >> > (phasefield, s, nx, ny, nz);
		cuSynchronize();
		//y=0
		dim3 dimGrid_y((nx + dimBlock.x - 1) / dimBlock.x, (nz + dimBlock.z - 1) / dimBlock.z);
		SetScalarFieldBoundary_y << <dimGrid_y, dimBlock >> > (phasefield, s, nx, ny, nz);
		cuSynchronize();
		//z=0
		dim3 dimGrid_z((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);
		SetScalarFieldBoundary_z << <dimGrid_z, dimBlock >> > (phasefield, s, nx, ny, nz);
		cuSynchronize();
		//xz=0
		dim3 dimGrid_xz((nx + dimBlock.x - 1) / dimBlock.x);
		SetScalarFieldBoundary_xz << <dimGrid_xz, dimBlock >> > (phasefield, nx, ny, nz);
		cuSynchronize();
		//yz=0
		dim3 dimGrid_yz((ny + dimBlock.y - 1) / dimBlock.y);
		SetScalarFieldBoundary_yz << <dimGrid_yz, dimBlock >> > (phasefield, nx, ny, nz);
		cuSynchronize();
		//xy=0
		dim3 dimGrid_xy((nz + dimBlock.z - 1) / dimBlock.z);
		SetScalarFieldBoundary_xy << <dimGrid_xy, dimBlock >> > (phasefield, nx, ny, nz);
		cuSynchronize();
		//dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		//P_SetScalarFieldBoundary_xyz << <dimGrid, dimBlock >> > (m_cuda_phasefield0, nx, ny, nz);
		//cuSynchronize();

	}




	/*
	date:2019/11/14
	author:@wdy
	describe:phasefield solver
	*/


	void PhaseField::PF_solver(float dt)
	{
		//float h = samplingDistance;

		//float w = 1.0f*h;//fro diffuse parameter
		//float gamma = 1.0f;
		//float ceo1 = 1.0f*gamma / h;		//for smoothing
		//float ceo2 = 1.5f*gamma*w / h / h;	//for sharping
		//float dif2 = (ceo1 + diff / h / h)*dt;
		////float a = (ceo1 + dif2 / h / h)*dt;
		//float c = 1.0f + 6.0f*dif2;
		//

		//
		//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		//dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		//K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield, m_cuda_phasefield0, nx, ny, nz);
		//PF_setScalarFieldBoundary(m_cuda_phasefield,true);

		////Kenel_AdvectForward << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, substep);
		////adection
		//P_AdvectWENO1rd << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, dt);
		//cuSynchronize();
		//K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, nx, ny, nz);

		//K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, nx, ny, nz);
		//PF_setScalarFieldBoundary(m_cuda_phasefield,true);

		//DiffusionPhiNorm << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_Norm, nx, ny, nz);
		//cuSynchronize();

		//DiffusionPhi << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_Norm, ceo2,nx, ny, nz, dt);
		//cuSynchronize();

		////K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefield, m_cuda_phasefield0, nx, ny, nz);
		//for (int it = 0; it < 20; it++)
		//{
		//	PF_setScalarFieldBoundary(m_cuda_phasefield, true);
		//	K_CopyGData << <dimGrid, dimBlock >> > (m_cuda_phasefieldcp, m_cuda_phasefield, nx, ny, nz);
		//	LinearSolve << <dimGrid, dimBlock >> > (m_cuda_phasefield0, m_cuda_phasefield, m_cuda_phasefieldcp,nx, ny, nz,c);
		//	cuSynchronize();
		//}
	}







	//__global__ void P_lDivergence(Grid1f Velu_x , Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz, float dt)
	//{
	//	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//	int k = blockIdx.z * blockDim.z + threadIdx.z;



	//	float totalDivergence;
	//	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < Velw_z.nz - 1)
	//	{
	//		totalDivergence += abs((Velu_x(i + 1, j, k) - Velu_x(i, j, k)) + (Velv_y(i, j + 1, k) - Velv_y(i, j, k)) + (Velw_z(i, j, k + 1) - Velw_z(i, j, k)));
	//	}
	//	if (totalDivergence>1)
	//	printf("散度=%f", totalDivergence);

	//}


	__global__ void P_lDivergenceVelecity(Grid1f Velu_x, Grid1f Velv_y, Grid1f Velw_z, int nx, int ny, int nz, float dt, float t)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		float T = 4.0f;

		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < Velw_z.nz - 1)
		{
			if (t < T && t > 0.0f)
			{
				float x = (i + 0.5f) / (float)nx;
				float y = (j + 0.5f) / (float)ny;
				if (t < T)
				{
					Velu_x(i + 1, j, k) = -2.0f*sin(M_PI*y)*cos(M_PI*y)*sin(M_PI*x)*sin(M_PI*x)*cos(t*M_PI / T);
					Velv_y(i, j + 1, k) = 2.0f*sin(M_PI*x)*cos(M_PI*x)*sin(M_PI*y)*sin(M_PI*y)*cos(t*M_PI / T);
					//Velv_y(i, j, k+1) = 2.0f*sin(M_PI*x)*cos(M_PI*x)*sin(M_PI*y)*sin(M_PI*y)*cos(t*M_PI / T);
				}
			}

		}
	}

	/*
	2019/11/13
	author@wdy
	describe:N-S equation solver
	*/
	void PhaseField::NS_solver(float dt, float t)
	{
		//cout << m_cuda_Velu.nx << endl;
		//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		//dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		//P_ApplyGravityForce << < dimGrid, dimBlock >> > (m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, dt);
		//cuSynchronize();

		////Interpolation from Boundary to Center
		//P_InterpolateVelocity << < dimGrid, dimBlock >> > (m_cuda_Veluc0, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz);
		//cuSynchronize();
		////Semi-Lagrangian Advection
		//P_AdvectionVelocity << < dimGrid, dimBlock >> > (m_cuda_Veluc, m_cuda_Veluc0, nx, ny, nz, dt);
		//cuSynchronize();
		////Interpolation from Center to Boundary
		//P_InterpolatedVelocity << < dimGrid, dimBlock >> > (m_cuda_Veluc, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, dt);
		//cuSynchronize();

		//dim3 dimGrid_x((ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		//P_SetU << < dimGrid_x, dimBlock >> > (m_cuda_Velu, nx, ny, nz);
		//dim3 dimGrid_y((nx + dimBlock.x - 1) / dimBlock.x, (nz + dimBlock.z - 1) / dimBlock.z);
		//P_SetV << < dimGrid_y, dimBlock >> > (m_cuda_Velv, nx, ny, nz);
		//dim3 dimGrid_z((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);
		//P_SetW << < dimGrid_z, dimBlock >> > (m_cuda_Velw, nx, ny, nz);

		//P_PrepareForProjection << < dimGrid, dimBlock >> > (m_cuda_CoefMatrix, m_cuda_RHS, m_cuda_phasefield0, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, dt);
		//cuSynchronize();
		//m_cuda_Pressure.cudaClear();
		////雅克比迭代求解压力
		//for (int i = 0; i < 300; i++)
		//{
		//	K_CopyGData << < dimGrid, dimBlock >> > (m_cuda_BufPressure, m_cuda_Pressure, nx, ny, nz);
		//	P_Projection << < dimGrid, dimBlock >> > (m_cuda_Pressure, m_cuda_BufPressure, m_cuda_CoefMatrix, m_cuda_RHS, nx, ny, nz);
		//	cuSynchronize();
		//}

		//P_UpdateVelocity_U << < dimGrid, dimBlock >> > (m_cuda_Velu, m_cuda_Pressure, m_cuda_phasefield0, nx, ny, nz, dt);
		//cuSynchronize();
		//P_UpdateVelocity_V << < dimGrid, dimBlock >> > (m_cuda_Velv, m_cuda_Pressure, m_cuda_phasefield0, nx, ny, nz, dt);
		//cuSynchronize();
		//P_UpdateVelocity_W << < dimGrid, dimBlock >> > (m_cuda_Velw, m_cuda_Pressure, m_cuda_phasefield0, nx, ny, nz, dt);
		//cuSynchronize();



		//P_lDivergenceVelecity << < dimGrid, dimBlock >> > (m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, substep,t);
		//cuSynchronize();
		//P_lDivergence << < dimGrid, dimBlock >> > (m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz, dt);
		//cuSynchronize();

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