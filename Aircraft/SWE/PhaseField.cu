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
#include <cufft.h>
#include "gl_utilities.h"
#include <time.h>
#include "PhaseField.h"
//#include"Ocean.h"
#include <iostream>

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

//float m_virtualGridSize = 0.1;
float samplingDistance = 0.02f;
/*
date:2019/11/14
author:@wdy
describe:Data copy on GPU
*/
__global__ void K_CopyData(float* dst, float* src, int nx, int ny, int nz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int index;
	if (i >= nx) return;
	if (j >= ny) return;
	if (k >= nz) return;
	index = i + j*nx + k*nx*ny;
	dst[index] = src[index];
}

/*******************************************************************************/
/***********************************Init_condition******************************/
/*******************************************************************************/
PhaseField::PhaseField(int size, float patchLength)
{
	//m_patch_length = patchLength;
	m_realGridSize = patchLength / size;

	m_fft_size = 128;

	simulatedRegionLenght = size;
	simulatedRegionWidth = size;
	simulatedRegionHeight = size;

	AllocateMemoery(m_fft_size, m_fft_size, m_fft_size);


}

PhaseField::~PhaseField(void)
{
}

void PhaseField::initialize()
{
	
	initDynamicRegion();
}



__global__ void C_InitDynamicRegion(float4* moveRegion, rgb* color1, int nx, int ny, int nz)
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

		vertex vij;
		vij.x = (float)i*m_virtualGridSize;
		vij.y = (float)0;
		vij.z = (float)j*m_virtualGridSize;
		vij.w = (float)1;

		moveRegion[index] = vij;
		color1[index] = make_uchar4(0, 120, 0, 220);
		//printf("%f", moveRegion[index].x);
	}
}

__global__ void C_InitPosition(float4* position, rgb* color, int nx, int ny, int nz)
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
		position[index] = make_float4(x, y, z, 0);
		color[index] = make_uchar4(0, 120, 120, 220);
	}
}


__global__ void C_PhaseField(float* phasefield, int nx, int ny, int nz)
{
	int index;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{
		index = i + j*nx + k*nx*ny;
		if (i > 5 && i < 35 && j > 5 && j < 100&&k>5&&k<35)
		{
			phasefield[index] = 1.0f;
		}
		else
		{
			phasefield[index] = 0.0f;
		}
	}
}


__global__ void C_PhaseChange(float* phasefield, rgb* color, int nx, int ny, int nz)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int index;

	if (i >= 0 && i < nx  && j >= 0 && j < ny && k >= 0 && k < nz)
	{
		index = i + j*nx + k*nx*ny;
		if (phasefield[index] == 0)
		{
			color[index] = make_uchar4(255, 255, 255, 0);
		}
		if (phasefield[index] == 1)
		{
			color[index] = make_uchar4(0, 0, 120, 220);
		}
		if (phasefield[index] > 0 && phasefield[index] < 1)
		{
			color[index] = make_uchar4(0, 120, 0, 220);
		}
	}
}

void PhaseField::initDynamicRegion()
{
	int extNx = simulatedRegionLenght;
	int extNy = simulatedRegionWidth;
	int extNz = simulatedRegionHeight;


	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
	dim3 dimGrid1((extNx + dimBlock.x - 1) / dimBlock.x, (extNy + dimBlock.y - 1) / dimBlock.y, (extNz + dimBlock.z - 1) / dimBlock.z);
	C_InitDynamicRegion << < dimGrid1, dimBlock >> > (m_SimulationRegion, m_SimulationRegionColor, extNx, extNy, extNz);
	synchronCheck;


	//dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
	C_InitPosition << < dimGrid, dimBlock >> > (m_cuda_position, m_cuda_color, nx, ny, nz);
	synchronCheck;

	C_PhaseField << <dimGrid, dimBlock >> > (m_cuda_phasefield, nx, ny, nz);
	synchronCheck;

	C_PhaseChange << <dimGrid, dimBlock >> > (m_cuda_phasefield, m_cuda_color, nx, ny, nz);
	synchronCheck;

}




__global__ void C_moveSimulationRegion(float4* moveRegion, int nx, int ny, int nz,int xstep,int ystep)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int index;

	if (i >= 0 && i < nx  && j >= 0 && j < ny && k >= 0 && k < nz)
	{

		index = i + j*nx + k*nx*ny;

		moveRegion[index].x += xstep*0.05;
		moveRegion[index].y += 0;
		moveRegion[index].z += ystep*0.05;
		moveRegion[index].w += 0;

	}
}

void PhaseField::moveSimulationRegion(int dx,int dy)
{
	//int eNx = simulatedRegionLenght;
	//int eNy = simulatedRegionWidth;
	//int eNz = simulatedRegionHeight;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((simulatedRegionLenght + dimBlock.x - 1) / dimBlock.x, (simulatedRegionWidth + dimBlock.y - 1) / dimBlock.y, (simulatedRegionHeight + dimBlock.z - 1) / dimBlock.z);
	C_moveSimulationRegion << < dimGrid, dimBlock >> > (m_SimulationRegion, simulatedRegionLenght, simulatedRegionWidth, simulatedRegionHeight,dx,dy);
	synchronCheck;

	m_simulatedOriginX += dx;
	m_simulatedOriginY += dy;
	cudaGraphicsUnmapResources(1, &SimulationRegion_resource, 0);
	cudaGraphicsUnmapResources(1, &SimulationRegionColor_resource, 0);

}



/*
date:2019/12/07
author:@wdy
describe:allocate memoery for all variable
*/
void PhaseField::AllocateMemoery(int _nx, int _ny, int _nz)
{

	//m_simulatedOriginX = 0;
	//m_simulatedOriginY = 0;

	nx = _nx;//计算区域-流体部分
	ny = _ny;
	nz = _nz;
	cudaError_t error;

	simulationRegitionSize=simulatedRegionLenght*simulatedRegionWidth*simulatedRegionHeight;
	simulationSize = nx*ny*nz;
	//for phasefield equation
	//仿真区域位置及颜色
	cudaCheck(cudaMalloc(&m_SimulationRegion, simulationRegitionSize * sizeof(float4)));
	cudaCheck(cudaMalloc(&m_SimulationRegionColor, simulationRegitionSize * sizeof(rgb)));

	cudaCheck(cudaMalloc(&m_cuda_position, simulationSize * sizeof(float4)));
	cudaCheck(cudaMalloc(&m_cuda_phasefield, simulationSize * sizeof(float)));
	cudaCheck(cudaMalloc(&m_cuda_color, nx* ny * sizeof(rgb)));
	//for N-S equation
	cudaCheck(cudaMalloc(&m_cuda_Velu, (nx + 1)*ny*nz * sizeof(float)));
	cudaCheck(cudaMalloc(&m_cuda_Velv, nx*(ny + 1)*nz * sizeof(float)));
	cudaCheck(cudaMalloc(&m_cuda_Velw, nx*ny*(nz + 1) * sizeof(float)));
	cudaCheck(cudaMalloc(&m_cuda_Veluc, simulationSize * sizeof(float3)));
	cudaCheck(cudaMalloc(&m_cuda_Veluc0, simulationSize * sizeof(float3)));
	cudaCheck(cudaMalloc(&m_cuda_CoefMatrix, simulationSize * sizeof(float7)));
	cudaCheck(cudaMalloc(&m_cuda_RHS, simulationSize * sizeof(float)));
	cudaCheck(cudaMalloc(&m_cuda_Pressure, simulationSize * sizeof(float)));
	cudaCheck(cudaMalloc(&m_cuda_BufPressure, simulationSize * sizeof(float)));
	cudaCheck(cudaMalloc(&m_cuda_Veluc1, simulationSize * sizeof(float3)));
	


	size_t size = simulationSize * sizeof(float3);
	size_t size1 = simulationRegitionSize * sizeof(float4);

	//仿真区域位置及颜色
	glGenBuffers(1, &SimulationRegion_bufferObj);
	glBindBuffer(GL_ARRAY_BUFFER, SimulationRegion_bufferObj);
	glBufferData(GL_ARRAY_BUFFER, simulationRegitionSize * sizeof(float4), NULL, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&SimulationRegion_resource, SimulationRegion_bufferObj, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &SimulationRegion_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_SimulationRegion, &size1, SimulationRegion_resource);

	glGenBuffers(1, &SimulationRegionColor_bufferObj);
	glBindBuffer(GL_ARRAY_BUFFER, SimulationRegionColor_bufferObj);
	glBufferData(GL_ARRAY_BUFFER, simulationRegitionSize * sizeof(rgb), NULL, GL_STATIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&SimulationRegionColor_resource, SimulationRegionColor_bufferObj, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &SimulationRegionColor_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_SimulationRegionColor, &size1, SimulationRegionColor_resource);

	

	//粒子位置、相场及颜色
	glGenBuffers(1, &Initpos_bufferObj);
	glBindBuffer(GL_ARRAY_BUFFER, Initpos_bufferObj);
	glBufferData(GL_ARRAY_BUFFER, simulationSize * sizeof(float4), NULL, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&Initpos_resource, Initpos_bufferObj, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &Initpos_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_cuda_position, &size, Initpos_resource);

	glGenBuffers(1, &PhaseField_bufferObj);
	glBindBuffer(GL_ARRAY_BUFFER, PhaseField_bufferObj);
	glBufferData(GL_ARRAY_BUFFER, simulationSize * sizeof(float), NULL, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&PhaseField_resource, PhaseField_bufferObj, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &PhaseField_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_cuda_phasefield, &size, PhaseField_resource);

	glGenBuffers(1, &Color_bufferObj);
	glBindBuffer(GL_ARRAY_BUFFER, Color_bufferObj);
	glBufferData(GL_ARRAY_BUFFER, simulationSize * sizeof(rgb), NULL, GL_STATIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&Color_resource, Color_bufferObj, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &Color_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_cuda_color, &size, Color_resource);


	//速度
	glGenBuffers(3, &Velocity_bufferObj[0]);
	glBindBuffer(GL_ARRAY_BUFFER, Velocity_bufferObj[0]);
	glBufferData(GL_ARRAY_BUFFER, (nx + 1)*ny*nz * sizeof(float), NULL, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&Velu_resource, Velocity_bufferObj[0], cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &Velu_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_cuda_Velu, &size, Velu_resource);

	glBindBuffer(GL_ARRAY_BUFFER, Velocity_bufferObj[1]);
	glBufferData(GL_ARRAY_BUFFER, nx*(ny + 1)*nz * sizeof(float), NULL, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&Velv_resource, Velocity_bufferObj[1], cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &Velv_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_cuda_Velv, &size, Velv_resource);

	glBindBuffer(GL_ARRAY_BUFFER, Velocity_bufferObj[2]);
	glBufferData(GL_ARRAY_BUFFER, nx*ny*(nz + 1) * sizeof(float), NULL, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&Velw_resource, Velocity_bufferObj[2], cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &Velw_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_cuda_Velw, &size, Velw_resource);

}


__global__ void C_Velecity(float3* Velu_c, float* Velu_x, float* Velu_y, float* Velu_z, int nx, int ny, int nz)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;


	int index;
	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{
		index = i + j*nx + k*nx*ny;
		Velu_c[index].x= 0.5*(Velu_x[index-1]+ Velu_x[index]);
		Velu_c[index].y= 0.5*(Velu_y[index - nx] + Velu_y[index]);
		Velu_c[index].z= 0.5*(Velu_z[index - nx*ny] + Velu_z[index]);
	}
}


__global__ void C_updatePosition(float4* position, float3* Velu_c, int nx, int ny, int nz,float dt)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;


	int index;
	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{
		index = i + j*nx + k*nx*ny;
		position[index].x += Velu_c[index].x*dt;
		position[index].y += Velu_c[index].y*dt;
		position[index].z += Velu_c[index].z*dt;
	}
}
void PhaseField::updatePosition(float dt)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
	C_updatePosition << <dimGrid, dimBlock >> > (m_cuda_position, m_cuda_Veluc1, nx, ny, nz, dt);
	synchronCheck;
	cudaGraphicsUnmapResources(1, &Initpos_resource, 0);
	cudaGraphicsUnmapResources(1, &Color_resource, 0);
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
		float substep = CFL();
		if (elipse + substep > dt)
		{
			substep = dt - elipse;
		}
		cout << "*********Substep: " << substep << " *********" << endl;
		t_start = clock();

		//N-S solver
		NS_solver(m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, m_cuda_phasefield, substep);
		t_start = clock();


		//m_cuda_phasefield0 = m_cuda_phasefield;
		K_CopyData << <dimGrid, dimBlock >> > (m_cuda_phasefield, m_cuda_phasefield0, nx, ny, nz);
		//PF solver
		PF_solver(m_cuda_phasefield, m_cuda_phasefield0, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, substep);
		C_Velecity << <dimGrid, dimBlock >> > (m_cuda_Veluc1, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz);
		//C_updatePosition << <dimGrid, dimBlock >> > (m_cuda_position, m_cuda_Veluc1, nx, ny, nz, substep);


		cudaGraphicsUnmapResources(1, &PhaseField_resource, 0);
		cudaGraphicsUnmapResources(1, &Velu_resource, 0);
		cudaGraphicsUnmapResources(1, &Velv_resource, 0);
		cudaGraphicsUnmapResources(1, &Velw_resource, 0);

		t_end = clock();
		cout << "Advect Time: " << t_end - t_start << endl;
		elipse += substep;
	}
	cout << dt << endl;
	t += dt;
	simItor = 0;
	clock_t total_end = clock();
	cout << "Total Cost " << total_end - total_start << " million seconds!" << endl;
	//if (simItor*dt > 4.01f)
	//{
	//	exit(0);
	//}
	//simItor++;

}

__global__ void C_CFL(float max_velu, float samplingDistance, float* Velu_x, float* Velv_y, float* Velw_z, int nx, int ny, int nz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	//float max_velu;
	float maxvel = 0.0f;
	if (i >= 0 && i < (nx + 1)*ny*nz)
	{
		maxvel = max(maxvel, abs(Velu_x[i]));
	}
	if (j >= 0 && j < nx*(ny + 1)*nz)
	{
		maxvel = max(maxvel, abs(Velv_y[j]));
	}
	if (k >= 0 && k < nx*ny*(nz + 1))
	{
		maxvel = max(maxvel, abs(Velw_z[k]));
	}
	if (maxvel < EPSILON)
		maxvel = 1.0f;
	max_velu = samplingDistance / maxvel;
}
/*
date:2019/11/14
author:@wdy
describe:CFL condition
*/
#define INNERINDEX(m,n,l) (m-1)*(ny-2)*(nz-2)+(n-1)*(nz-2)+l-1
float PhaseField::CFL()
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);

	float max_velu;
	C_CFL << <dimGrid, dimBlock >> > (max_velu, samplingDistance, m_cuda_Velu, m_cuda_Velv, m_cuda_Velw, nx, ny, nz);
	cuSynchronize();
	return max_velu;
}





/*******************************************************************************/
/********************************PhaseField_Solver******************************/
/*******************************************************************************/

__global__ void P_AdvectWENO1rd(float*phaseField, float* phaseField0, float* Velu_x, float* Velv_y, float* Velw_z, int nx, int ny, int nz, float dt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;


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
			index0 = ix0 + iy0*nx + iz0*nx*ny;
			index1 = ix1 + iy1*nx + iz1*nx*ny;
			//u_mid = u(i + 1, j, k);
			u_mid = Velu_x[(i + 1) + j*nx + k*nx*ny];
			if (u_mid > 0.0f)
			{
				//c_mid = d0(ix0, iy0, iz0);
				c_mid = phaseField0[index0];
			}
			else
			{
				c_mid = phaseField0[index1];
			}
			dc = dt*invh*c_mid*u_mid;
			atomicAdd(&(phaseField[index0]), -dc);
			atomicAdd(&(phaseField[index1]), dc);
		}

		//j and j+1
		ix0 = i; iy0 = j;   iz0 = k;
		ix1 = i; iy1 = j + 1; iz1 = k;
		if (iy1 < ny - 1)
		{
			index0 = ix0 + iy0*nx + iz0*nx*ny;
			index1 = ix1 + iy1*nx + iz1*nx*ny;

			u_mid = Velv_y[i + (j + 1)*nx + k*nx*ny];
			if (u_mid > 0.0f)
			{
				c_mid = phaseField0[index0];
			}
			else
			{
				c_mid = phaseField0[index1];
			}
			dc = dt*invh*c_mid*u_mid;
			atomicAdd(&(phaseField[index0]), -dc);
			atomicAdd(&(phaseField[index1]), dc);
		}

		ix0 = i; iy0 = j;   iz0 = k;
		ix1 = i; iy1 = j; iz1 = k + 1;
		if (iz1 < nz - 1)
		{
			index0 = ix0 + iy0*nx + iz0*nx*ny;
			index1 = ix1 + iy1*nx + iz1*nx*ny;

			u_mid = Velw_z[i + j*nx + (k + 1)*nx*ny];
			if (u_mid > 0.0f)
			{
				c_mid = phaseField0[index0];
			}
			else
			{
				c_mid = phaseField0[index1];
			}
			dc = dt*invh*c_mid*u_mid;
			atomicAdd(&(phaseField[index0]), -dc);
			atomicAdd(&(phaseField[index1]), dc);
		}
	}
}
/*
date:2019/11/14
author:@wdy
describe:phasefield solver
*/

void PhaseField::PF_solver(float* phaseField, float* phaseField0, float* Velu_x, float* Velv_y, float* Velw_z, float dt)
{
	PF_setScalarFieldBoundary(phaseField0, true);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
	//adection
	P_AdvectWENO1rd << <dimGrid, dimBlock >> > (phaseField, phaseField0, Velu_x, Velv_y, Velw_z, nx, ny, nz, dt);
	cuSynchronize();


}


__global__ void P_SetScalarFieldBoundary_x(float* field, float s, int nx, int ny, int nz)
{

	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{
		field[0 + j*nx + k*nx*ny] = s*field[1 + j*nx + k*nx*ny];
		field[(nx - 1) + j*nx + k*nx*ny] = s*field[(nx - 2) + j*nx + k*nx*ny];
		//field(0, j, k) = s * field(1, j, k);
		//field(nx - 1, j, k) = s * field(nx - 2, j, k);
	}
}

__global__ void P_SetScalarFieldBoundary_y(float* field, float s, int nx, int ny, int nz)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i >= 1 && i < nx - 1 && k >= 1 && k < nz - 1)
	{

		field[i + (0 * nx) + k*nx*ny] = s*field[i + (1 * nx) + k*nx*ny];
		field[i + (ny - 1)*nx + k*nx*ny] = s*field[i + (ny - 2)*nx + k*nx*ny];
		//field(i, 0, k) = s * field(i, 1, k);
		//field(i, ny - 1, k) = s * field(i, ny - 2, k);
	}
}

__global__ void P_SetScalarFieldBoundary_z(float* field, float s, int nx, int ny, int nz)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
	{
		field[i + j*nx + (0 * nx*ny)] = s*field[i + j*nx + (1 * nx*ny)];
		field[i + j*nx + (nz - 1)*nx*ny] = s*field[i + j*nx + (nz - 2)*nx*ny];
		//field(i, j, 0) = s * field(i, j, 1);
		//field(i, j, nz - 1) = s * field(i, j, nz - 2);
	}
}


__global__ void P_SetScalarFieldBoundary_yz(float* field, int nx, int ny, int nz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 0 && i < nx)
	{
		field[i + 0 * nx + 0 * nx*ny] = 0.5*(field[i + 1 * nx + 0 * nx*ny] + field[i + 0 * nx + 1 * nx*ny]);
		field[i + (ny - 1)*nx + 0 * nx*ny] = 0.5*(field[i + (ny - 2)*nx + 0 * nx*ny] + field[i + (ny - 1)*nx + 1 * nx*ny]);
		field[i + 0 * nx + (nz - 1)*nx*ny] = 0.5*(field[i + 1 * nx + (nz - 1)*nx*ny] + field[i + 0 * nx + (nz - 2)*nx*ny]);
		field[i + (ny - 1)*nx + (nz - 1)*nx*ny] = 0.5*(field[i + (ny - 1)*nx + (ny - 2)*nx*ny] + field[i + (ny - 2)*nx + (nz - 1)*nx*ny]);
		//field(i, 0, 0) = 0.5f*(field(i, 1, 0) + field(i, 0, 1));
		//field(i, Ny - 1, 0) = 0.5f*(field(i, Ny - 2, 0) + field(i, Ny - 1, 1));
		//field(i, 0, Nz - 1) = 0.5f*(field(i, 1, Nz - 1) + field(i, 0, Nz - 2));
		//field(i, Ny - 1, Nz - 1) = 0.5f*(field(i, Ny - 1, Nz - 2) + field(i, Ny - 2, Nz - 1));
	}
}

__global__ void P_SetScalarFieldBoundary_xz(float* field, int nx, int ny, int nz)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= 1 && j < ny - 1)
	{
		field[0 + j*nx + 0 * nx*ny] = 0.5*(field[1 + j*nx + 0 * nx*ny] + field[0 + j*nx + 1 * nx*ny]);
		field[0 + j*nx + (nz - 1)*nx*ny] = 0.5*(field[1 + j*nx + (nz - 1)*nx*ny] + field[0 + j*nx + (nz - 2)*nx*ny]);
		field[(nx - 1) + j*nx + 0 * nx*ny] = 0.5*(field[(nx - 2) + j*nx + 0 * nx*ny] + field[(nx - 2) + j*nx + 1 * nx*ny]);
		field[(nx - 1) + j*nx + (nz - 1)*nx*ny] = 0.5*(field[(nx - 2) + j*nx + (nz - 1)*nx*ny] + field[(nx - 1) + j*nx + (nz - 2)*nx*ny]);

		//field(0, j, 0) = 0.5f*(field(1, j, 0) + field(0, j, 1));
		//field(0, j, Nz - 1) = 0.5f*(field(1, j, Nz - 1) + field(0, j, Nz - 2));
		//field(Nx - 1, j, 0) = 0.5f*(field(Nx - 2, j, 0) + field(Nx - 2, j, 1));
		//field(Nx - 1, j, Nz - 1) = 0.5f*(field(Nx - 2, j, Nz - 1) + field(Nx - 1, j, Nz - 2));
	}
}

__global__ void P_SetScalarFieldBoundary_xy(float* field, int nx, int ny, int nz)
{
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (k >= 1 && k < nz - 1)
	{
		field[0 + 0 * nx + k*nx*ny] = 0.5*(field[1 + 0 * nx + k*nx*ny] + field[0 + 1 * nx + k*nx*ny]);
		field[(nx - 1) + 0 * nx + k*nx*ny] = 0.5*(field[(nx - 2) + 0 * nx + k*nx*ny] + field[(nx - 1) + 1 * nx + k*nx*ny]);
		field[0 + (ny - 1)*nx + k*nx*ny] = 0.5*(field[1 + (ny - 1)*nx + k*nx*ny] + field[0 + (ny - 2)*nx + k*nx*ny]);
		field[(nx - 1) + (ny - 1)*nx + k*nx*ny] = 0.5*(field[(nx - 2) + (ny - 1)*nx + k*nx*ny] + field[(nx - 1) + (ny - 2)*nx + k*nx*ny]);
		//field(0, 0, k) = 0.5f*(field(1, 0, k) + field(0, 1, k));
		//field(Nx - 1, 0, k) = 0.5f*(field(Nx - 2, 0, k) + field(Nx - 1, 1, k));
		//field(0, Ny - 1, k) = 0.5f*(field(1, Ny - 1, k) + field(0, Ny - 2, k));
		//field(Nx - 1, Ny - 1, k) = 0.5f*(field(Nx - 2, Ny - 1, k) + field(Nx - 1, Ny - 2, k));
	}
}


/*
2019/11/14
author@wdy
describe:Setting field boundary
*/
void PhaseField::PF_setScalarFieldBoundary(float* phaseField, bool postive)
{
	float s = postive ? 1.0f : -1.0f;
	//computer
	//x=0
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid_x((ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
	P_SetScalarFieldBoundary_x << <dimGrid_x, dimBlock >> > (phaseField, s, nx, ny, nz);
	cuSynchronize();
	//y=0
	dim3 dimGrid_y((nx + dimBlock.x - 1) / dimBlock.x, (nz + dimBlock.z - 1) / dimBlock.z);
	P_SetScalarFieldBoundary_y << <dimGrid_y, dimBlock >> > (phaseField, s, nx, ny, nz);
	cuSynchronize();
	//z=0
	dim3 dimGrid_z((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);
	P_SetScalarFieldBoundary_z << <dimGrid_z, dimBlock >> > (phaseField, s, nx, ny, nz);
	cuSynchronize();
	//xz=0
	dim3 dimGrid_xz((nx + dimBlock.x - 1) / dimBlock.x);
	P_SetScalarFieldBoundary_xz << <dimGrid_xz, dimBlock >> > (phaseField, nx, ny, nz);
	cuSynchronize();
	//yz=0
	dim3 dimGrid_yz((ny + dimBlock.y - 1) / dimBlock.y);
	P_SetScalarFieldBoundary_yz << <dimGrid_yz, dimBlock >> > (phaseField, nx, ny, nz);
	cuSynchronize();
	//xy=0
	dim3 dimGrid_xy((nz + dimBlock.z - 1) / dimBlock.z);
	P_SetScalarFieldBoundary_xy << <dimGrid_xy, dimBlock >> > (phaseField, nx, ny, nz);
	cuSynchronize();
}







/*******************************************************************************/
/*************************************N-S_Solver*********************************/
/*******************************************************************************/
/*
2019/10/27
author@wdy
describe: Apply gravity
*/
__global__ void P_ApplyGravityForce(float* Velu, float* Velv, float* Velw, int nx, int ny, int nz, float dt)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int index;
	if (i >= 1 && i < nx - 1 && j >= 2 && j < ny - 2 && k >= 1 && k < nz - 1)
	{
		index = i + j*nx + k*nx*ny;
		Velu[index] += 0.0f;
		Velv[index] += dt*g;
		Velw[index] += 0.0f;
	}
}


/*
2019/10/27
author@wdy
describe: Advection velecity
*/
__global__ void P_InterpolateVelocity(float3* Velu_c, float* Velu_x, float* Velv_y, float* Velw_z, int nx, int ny, int nz)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int index;
	float3 vel_ijk;
	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{
		index = i + j*nx + k*nx*ny;
		Velu_c[index].x = 0.5f*(Velu_x[index] + Velu_x[index + 1]);
		Velu_c[index].y = 0.5f*(Velv_y[index] + Velv_y[index + nx]);
		Velu_c[index].z = 0.5f*(Velw_z[index] + Velw_z[index + nx*ny]);
	}
}

__global__ void P_AdvectionVelocity(float3* Velu_c, float3* Velu_c0, int nx, int ny, int nz, float dt)
{
	float h = 0.005f;
	float fx, fy, fz;
	int  ix, iy, iz;
	float w000, w100, w010, w001, w111, w011, w101, w110;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int index;
	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{
		index = i + j*nx + k*nx*ny;
		fx = i + dt*Velu_c0[index].x / h;
		fy = j + dt*Velu_c0[index].y / h;
		fz = k + dt*Velu_c0[index].z / h;

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
		atomicAdd(&Velu_c[ix + iy*nx + iz*nx*ny].x, Velu_c0[index].x * w000);
		atomicAdd(&Velu_c[(ix + 1) + iy*nx + iz*nx*ny].x, Velu_c0[index].x * w100);
		atomicAdd(&Velu_c[ix + (iy + 1)*nx + iz*nx*ny].x, Velu_c0[index].x * w010);
		atomicAdd(&Velu_c[ix + iy*nx + (iz + 1)*nx*ny].x, Velu_c0[index].x * w001);

		atomicAdd(&Velu_c[(ix + 1) + (iy + 1)*nx + (iz + 1)*nx*ny].x, Velu_c0[index].x * w111);
		atomicAdd(&Velu_c[ix + (iy + 1)*nx + (iz + 1)*nx*ny].x, Velu_c0[index].x * w011);
		atomicAdd(&Velu_c[(ix + 1) + iy*nx + (iz + 1)*nx*ny].x, Velu_c0[index].x * w101);
		atomicAdd(&Velu_c[(ix + 1) + (iy + 1)*nx + iz*nx*ny].x, Velu_c0[index].x * w110);

		//y direction
		atomicAdd(&Velu_c[ix + iy*nx + iz*nx*ny].y, Velu_c0[index].y * w000);
		atomicAdd(&Velu_c[(ix + 1) + iy*nx + iz*nx*ny].y, Velu_c0[index].y * w100);
		atomicAdd(&Velu_c[ix + (iy + 1)*nx + iz*nx*ny].y, Velu_c0[index].y * w010);
		atomicAdd(&Velu_c[ix + iy*nx + (iz + 1)*nx*ny].y, Velu_c0[index].y * w001);

		atomicAdd(&Velu_c[(ix + 1) + (iy + 1)*nx + (iz + 1)*nx*ny].y, Velu_c0[index].y * w111);
		atomicAdd(&Velu_c[ix + (iy + 1)*nx + (iz + 1)*nx*ny].y, Velu_c0[index].y * w011);
		atomicAdd(&Velu_c[(ix + 1) + iy*nx + (iz + 1)*nx*ny].y, Velu_c0[index].y * w101);
		atomicAdd(&Velu_c[(ix + 1) + (iy + 1)*nx + iz*nx*ny].y, Velu_c0[index].y * w110);

		//z direction
		atomicAdd(&Velu_c[ix + iy*nx + iz*nx*ny].z, Velu_c0[index].z * w000);
		atomicAdd(&Velu_c[(ix + 1) + iy*nx + iz*nx*ny].z, Velu_c0[index].z * w100);
		atomicAdd(&Velu_c[ix + (iy + 1)*nx + iz*nx*ny].z, Velu_c0[index].z * w010);
		atomicAdd(&Velu_c[ix + iy*nx + (iz + 1)*nx*ny].z, Velu_c0[index].z * w001);

		atomicAdd(&Velu_c[(ix + 1) + (iy + 1)*nx + (iz + 1)*nx*ny].z, Velu_c0[index].z * w111);
		atomicAdd(&Velu_c[ix + (iy + 1)*nx + (iz + 1)*nx*ny].z, Velu_c0[index].z * w011);
		atomicAdd(&Velu_c[(ix + 1) + iy*nx + (iz + 1)*nx*ny].z, Velu_c0[index].z * w101);
		atomicAdd(&Velu_c[(ix + 1) + (iy + 1)*nx + iz*nx*ny].z, Velu_c0[index].z * w110);
	}
}

__global__ void P_InterpolatedVelocity(float3* Velu_c, float* Velu_x, float* Velv_y, float* Velw_z, int nx, int ny, int nz, float dt)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;


	int index;
	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{
		index = i + j*nx + k*nx*ny;
		Velu_x[index] = 0.5f*(Velu_c[index].x + Velu_c[index + 1].x);
		Velv_y[index] = 0.5f*(Velu_c[index].y + Velu_c[index + nx].y);
		Velw_z[index] = 0.5f*(Velu_c[index].z + Velu_c[index + nx*ny].z);
	}
}



/*
2019/10/27
author@wdy
describe: Set boundary
*/
__global__ void P_SetU(float* Velu_x, int nx, int ny, int nz)
{
	/*	int i = blockDim.x * blockIdx.x + threadIdx.x;*/
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (j >= 0 && j < ny && k >= 0 && k < nz)
	{
		Velu_x[0 + j*nx + k*nx*ny] = 0.0f;
		Velu_x[1 + j*nx + k*nx*ny] = 0.0f;
		Velu_x[nx + j*nx + k*nx*ny] = 0.0f;
		Velu_x[(nx - 1) + j*nx + k*nx*ny] = 0.0f;
	}
}

__global__ void P_SetV(float* Velv_y, int nx, int ny, int nz)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int k = blockIdx.z * blockDim.z + threadIdx.z;


	if (i >= 0 && i < nx && k >= 0 && k < nz)
	{
		Velv_y[i + 0 * nx + k*nx*ny] = 0.0f;
		Velv_y[i + 1 * nx + k*nx*ny] = 0.0f;
		Velv_y[i + ny * nx + k*nx*ny] = 0.0f;
		Velv_y[i + (ny - 1) * nx + k*nx*ny] = 0.0f;
	}
}


__global__ void P_SetW(float* Velw_z, int nx, int ny, int nz)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i >= 0 && i < nx && j >= 0 && j < ny)
	{
		Velw_z[i + j * nx + 0 * nx*ny] = 0.0f;
		Velw_z[i + j * nx + 1 * nx*ny] = 0.0f;
		Velw_z[i + j * nx + nz*nx*ny] = 0.0f;
		Velw_z[i + j * nx + (nz - 1)*nx*ny] = 0.0f;
	}
}

/*
2019/10/27
author@wdy
describe: Solve divergence and  coefficient
*/
__global__ void P_PrepareForProjection(float7* coefMatrix, float* RHS, float* mass, float* Velu_x, float* Velv_y, float* Velw_z, int nx, int ny, int nz, float dt)
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
	int index;
	float div_ijk = 0.0f;
	float S = 0.9f;
	float7 A_ijk;

	A_ijk.a = 0.0f;
	A_ijk.x0 = 0.0f;
	A_ijk.x1 = 0.0f;
	A_ijk.y0 = 0.0f;
	A_ijk.y1 = 0.0f;
	A_ijk.z0 = 0.0f;
	A_ijk.z1 = 0.0f;
	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{
		index = i + j*nx + k*nx*ny;
		float m_ijk = mass[index];
		if (i < nx - 2) {
			float c = 0.5f*(m_ijk + mass[index + 1]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO2*c + RHO1*(1.0f - c));//分母是密度，c是phi
															  //term = term*hh;
			A_ijk.a += term;
			A_ijk.x1 += term;
		}
		div_ijk -= Velu_x[index + 1] / h;
		//left neighbour
		if (i > 1) {
			float c = 0.5f*(m_ijk + mass[index - 1]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO2*c + RHO1*(1.0f - c));
			//term = term*hh;
			A_ijk.a += term;
			A_ijk.x0 += term;
		}

		div_ijk += Velu_x[index] / h;

		//top neighbour
		if (j < ny - 2) {
			float c = 0.5f*(m_ijk + mass[index + nx]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO2*c + RHO1*(1.0f - c));
			//term = term*hh;
			A_ijk.a += term;
			A_ijk.y1 += term;
		}
		div_ijk -= Velv_y[index + nx] / h;

		//bottom neighbour
		if (j > 1) {
			float c = 0.5f*(m_ijk + mass[index - nx]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO2*c + RHO1*(1.0f - c));
			//term = term*hh;
			A_ijk.a += term;
			A_ijk.y0 += term;
		}
		div_ijk += Velv_y[index] / h;

		//far neighbour
		if (k < nz - 2) {
			float c = 0.5f*(m_ijk + mass[index + nx*ny]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO2*c + RHO1*(1.0f - c));
			//term = term*hh;
			A_ijk.a += term;
			A_ijk.z1 += term;

		}
		div_ijk -= Velw_z[index + nx*ny] / h;

		//near neighbour
		if (k > 1) {
			float c = 0.5f*(m_ijk + mass[index - nx*ny]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO2*c + RHO1*(1.0f - c));
			//term = term*hh;
			A_ijk.a += term;
			A_ijk.z0 += term;
		}
		div_ijk += Velw_z[index] / h;

		if (m_ijk > 1.0)
		{
			div_ijk += 0.5f*pow((m_ijk - 1.0f), 1.0f) / dt;
			//div_ijk += S*((mass(i + 1, j, k) - m_ijk)+ (mass(i - 1, j, k) - m_ijk)+ (mass(i, j + 1, k) - m_ijk)+ (mass(i, j - 1, k) - m_ijk) + (mass(i, j, k + 1) - m_ijk)+  (mass(i, j, k - 1) - m_ijk)) / m_ijk / dt;
		}

		coefMatrix[index] = A_ijk;
		RHS[index] = div_ijk;//度散
	}
}

/*
2019/10/27
author@wdy
describe:Solve pressure
*/
__global__ void P_Projection(float* pressure, float* bufPressure, float7* coefMatrix, float* RHS, int nx, int ny, int nz)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{
		int k0 = i + j*nx + k*nx*ny;
		float7 A_ijk = coefMatrix[k0];

		float a = A_ijk.a;
		float x0 = A_ijk.x0;
		float x1 = A_ijk.x1;
		float y0 = A_ijk.y0;
		float y1 = A_ijk.y1;
		float z0 = A_ijk.z0;
		float z1 = A_ijk.z1;
		float p_ijk;

		p_ijk = RHS[k0];
		if (i > 0) p_ijk += x0*bufPressure[k0 - 1];
		if (i < nx - 1) p_ijk += x1*bufPressure[k0 + 1];
		if (j > 0) p_ijk += y0*bufPressure[k0 - nx];
		if (j < ny - 1) p_ijk += y1*bufPressure[k0 + nx];
		if (k > 0) p_ijk += z0*bufPressure[k0 - nx*ny];
		if (k < nz - 1) p_ijk += z1*bufPressure[k0 + nx*ny];

		pressure[k0] = p_ijk / a;
	}

}


__global__ void P_UpdateVelocity_U(float* Velu_x, float* pressure, float* mass, int nx, int ny, int nz, float dt)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;


	int ni = nx;
	int nj = ny;
	int nk = nz;
	int nij = ni*nj;

	if (i >= 2 && i < (nx + 1) - 2 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
	{

		float h = 0.005f;
		int index = i + j*ni + k*ni*nj;
		float c = 0.5f*(mass[index - 1] + mass[index]);
		c = c > 1.0f ? 1.0f : c;
		c = c < 0.0f ? 0.0f : c;

		Velu_x[index] -= dt*(pressure[index] - pressure[index - 1]) / h / (c*RHO2 + (1.0f - c)*RHO1);
	}
}

__global__ void P_UpdateVelocity_V(float* Velv_y, float* pressure, float* mass, int nx, int ny, int nz, float dt)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int ni = nx;
	int nj = ny;
	int nk = nz;
	int nij = ni*nj;

	if (i >= 1 && i < nx - 1 && j >= 2 && j < (ny + 1) - 2 && k >= 1 && k < nz - 1)
	{

		float h = 0.005f;
		int index = i + j*ni + k*ni*nj;

		float c = 0.5f*(mass[index] + mass[index - ni]);
		c = c > 1.0f ? 1.0f : c;
		c = c < 0.0f ? 0.0f : c;
		Velv_y[index] -= dt*(pressure[index] - pressure[index - ni]) / h / (c*RHO2 + (1.0f - c)*RHO1);
	}
}

__global__ void P_UpdateVelocity_W(float* Velw_z, float* pressure, float* mass, int nx, int ny, int nz, float dt)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;


	int ni = nx;
	int nj = ny;
	int nk = nz;
	int nij = ni*nj;

	if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 2 && k < (nz + 1) - 2)
	{

		float h = 0.005f;
		int index = i + j*ni + k*ni*nj;

		float c = 0.5f*(mass[index] + mass[index - nij]);
		c = c > 1.0f ? 1.0f : c;
		c = c < 0.0f ? 0.0f : c;
		Velw_z[index] -= dt*(pressure[index] - pressure[index - nij]) / h / (c*RHO2 + (1.0f - c)*RHO1);
	}
}
/*
2019/11/13
author@wdy
describe:N-S equation solver
*/
void PhaseField::NS_solver(float* Velu_x, float* Velv_y, float* Velw_z, float* cudaPhase, float substep)
{

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
	P_ApplyGravityForce << < dimGrid, dimBlock >> > (Velu_x, Velv_y, Velw_z, nx, ny, nz, substep);
	cuSynchronize();

	//Interpolation from Boundary to Center
	P_InterpolateVelocity << < dimGrid, dimBlock >> > (m_cuda_Veluc0, Velu_x, Velv_y, Velw_z, nx, ny, nz);
	cuSynchronize();
	//Semi-Lagrangian Advection
	P_AdvectionVelocity << < dimGrid, dimBlock >> > (m_cuda_Veluc, m_cuda_Veluc0, nx, ny, nz, substep);
	cuSynchronize();
	//Interpolation from Center to Boundary
	P_InterpolatedVelocity << < dimGrid, dimBlock >> > (m_cuda_Veluc, Velu_x, Velv_y, Velw_z, nx, ny, nz, substep);
	cuSynchronize();


	dim3 dimGrid_x((ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
	P_SetU << < dimGrid_x, dimBlock >> > (Velu_x, nx, ny, nz);
	dim3 dimGrid_y((nx + dimBlock.x - 1) / dimBlock.x, (nz + dimBlock.z - 1) / dimBlock.z);
	P_SetV << < dimGrid_y, dimBlock >> > (Velv_y, nx, ny, nz);
	dim3 dimGrid_z((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);
	P_SetW << < dimGrid_z, dimBlock >> > (Velw_z, nx, ny, nz);


	P_PrepareForProjection << < dimGrid, dimBlock >> > (m_cuda_CoefMatrix, m_cuda_RHS, cudaPhase, Velu_x, Velv_y, Velw_z, nx, ny, nz, substep);
	cuSynchronize();


	//雅克比迭代求解压力
	for (int i = 0; i < 50; i++)
	{
		K_CopyData << < dimGrid, dimBlock >> > (m_cuda_BufPressure, m_cuda_Pressure, nx, ny, nz);
		P_Projection << < dimGrid, dimBlock >> > (m_cuda_Pressure, m_cuda_BufPressure, m_cuda_CoefMatrix, m_cuda_RHS, nx, ny, nz);
		cuSynchronize();
	}

	P_UpdateVelocity_U << < dimGrid, dimBlock >> > (Velu_x, m_cuda_Pressure, cudaPhase, nx, ny, nz, substep);
	cuSynchronize();
	P_UpdateVelocity_V << < dimGrid, dimBlock >> > (Velv_y, m_cuda_Pressure, cudaPhase, nx, ny, nz, substep);
	cuSynchronize();
	P_UpdateVelocity_W << < dimGrid, dimBlock >> > (Velw_z, m_cuda_Pressure, cudaPhase, nx, ny, nz, substep);
	cuSynchronize();
}



void PhaseField::display()
{
	glEnableClientState(GL_INDEX_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);


	glBindBuffer(GL_ARRAY_BUFFER, SimulationRegionColor_bufferObj);
	glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(rgb), 0);

	glBindBuffer(GL_ARRAY_BUFFER, SimulationRegion_bufferObj);
	glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDrawElements(GL_POINTS, 4 * (simulatedRegionLenght - 1)*(simulatedRegionWidth - 1)*(simulatedRegionHeight - 1), GL_UNSIGNED_INT, 0);
	glDisable(GL_BLEND);


	glBindBuffer(GL_ARRAY_BUFFER, Color_bufferObj);
	glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(rgb), 0);

	glBindBuffer(GL_ARRAY_BUFFER, Initpos_bufferObj);
	glVertexPointer(3, GL_FLOAT, sizeof(float4), 0);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDrawElements(GL_POINTS, 4 * (nx - 1)*(ny - 1)*(nz - 1), GL_UNSIGNED_INT, 0);
	glDisable(GL_BLEND);

	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_INDEX_ARRAY);

}