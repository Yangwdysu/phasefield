#if __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/glew.h>
#endif

#include <iostream>
#include "CapillaryWave.h"
#include "cuda_helper_math.h"
#include <cuda_gl_interop.h>  
#include <cufft.h>
#include "gl_utilities.h"
namespace WetBrush {
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define grid2Dwrite(array, x, y, pitch, value) array[(y)*pitch+x] = value
#define grid2Dread(array, x, y, pitch) array[(y)*pitch+x]

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

	texture<gridpoint, 2, cudaReadModeElementType> g_capillaryTexture;
	cudaChannelFormatDesc g_cpChannelDesc;

	__constant__ float GRAVITY = 9.83219f * 0.5f; //0.5f * Fallbeschleunigung

	CapillaryWave::CapillaryWave(int size, float patchLength)
	{
		m_patch_length = patchLength;
		m_realGridSize = patchLength / size;

		m_simulatedRegionWidth = size;
		m_simulatedRegionHeight = size;

		m_simulatedOriginX = 0;
		m_simulatedOriginY = 0;

		initialize();
	}

	CapillaryWave::~CapillaryWave()
	{
		cudaFree(m_device_grid);
		cudaFree(m_device_grid_next);
		cudaFree(m_height);
		cudaFree(m_source);
		cudaFree(m_weight);
		glDeleteTextures(1, &m_height_texture);
	}

	void CapillaryWave::initialize()
	{
		initDynamicRegion();

		initSource();
	}

	__global__ void C_InitDynamicRegion(gridpoint *grid, int gridwidth, int gridheight, int pitch, float level)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < gridwidth && y < gridheight)
		{
			gridpoint gp;
			gp.x = level;
			gp.y = 0.0f;
			gp.z = 0.0f;
			gp.w = 0.0f;

			grid2Dwrite(grid, x, y, pitch, gp);//���������ݴ���device_grid��
		}
	}

	void CapillaryWave::initDynamicRegion()
	{
		cudaError_t error;

		int extNx = m_simulatedRegionWidth + 2;
		int extNy = m_simulatedRegionHeight + 2;

		size_t pitch;
		cudaCheck(cudaMallocPitch(&m_device_grid, &pitch, extNx * sizeof(gridpoint), extNy));
		cudaCheck(cudaMallocPitch(&m_device_grid_next, &pitch, extNx * sizeof(gridpoint), extNy));

		cudaCheck(cudaMalloc((void **)&m_height, m_simulatedRegionWidth*m_simulatedRegionWidth * sizeof(float4)));

		gl_utility::createTexture(m_simulatedRegionWidth, m_simulatedRegionHeight, GL_RGBA32F, m_height_texture, GL_CLAMP_TO_BORDER, GL_LINEAR, GL_LINEAR, GL_RGBA, GL_FLOAT);
		cudaCheck(cudaGraphicsGLRegisterImage(&m_cuda_texture, m_height_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

		m_grid_pitch = pitch / sizeof(gridpoint);

		int x = (extNx + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (extNy + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		//init grid with initial values
		C_InitDynamicRegion << < blocksPerGrid, threadsPerBlock >> > (m_device_grid, extNx, extNy, m_grid_pitch, m_horizon);
		synchronCheck;

		//init grid_next with initial values
		C_InitDynamicRegion << < blocksPerGrid, threadsPerBlock >> > (m_device_grid_next, extNx, extNy, m_grid_pitch, m_horizon);
		synchronCheck;

		error = cudaThreadSynchronize();

		g_cpChannelDesc = cudaCreateChannelDesc<float4>();
		cudaCheck(cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint)));
	}

	__global__ void C_InitSource(
		float2* source,
		int patchSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			if (i < patchSize / 2 + 3 && i > patchSize / 2 - 3 && j < patchSize / 2 + 3 && j > patchSize / 2 - 3)
			{
				float2 uv = make_float2(1.0f);
				source[i + j*patchSize] = uv;
			}
		}
	}

	void CapillaryWave::initSource()
	{
		int sizeInBytes = m_simulatedRegionWidth* m_simulatedRegionHeight * sizeof(float2);

		cudaCheck(cudaMalloc(&m_source, sizeInBytes));
		cudaCheck(cudaMalloc(&m_weight, m_simulatedRegionWidth* m_simulatedRegionHeight * sizeof(float)));
		cudaCheck(cudaMemset(m_source, 0, sizeInBytes));

		int x = (m_simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);
		C_InitSource << < blocksPerGrid, threadsPerBlock >> > (m_source, m_simulatedRegionWidth);
		resetSource();
		synchronCheck;
	}

	__global__ void C_MoveSimulatedRegion(
		gridpoint *grid,
		int width,
		int height,
		int dx,
		int dy,
		int pitch,
		float horizon)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < width && j < height)
		{
			int gx = i + 1;
			int gy = j + 1;

			float4 gp = tex2D(g_capillaryTexture, gx, gy);
			float4 gp_init = make_float4(horizon, 0.0f, 0.0f, gp.w);

			int new_i = i - dx;
			int new_j = j - dy;


			gp = new_i < 0 || new_i >= width ? gp_init : gp;
			new_i = new_i % width;
			new_i = new_i < 0 ? width + new_i : new_i;

			gp = new_j < 0 || new_j >= height ? gp_init : gp;
			new_j = new_j % height;
			new_j = new_j < 0 ? height + new_j : new_j;

			grid2Dwrite(grid, new_i + 1, new_j + 1, pitch, gp);
		}
	}

	void CapillaryWave::moveDynamicRegion(int nx, int ny)
	{
		int extNx = m_simulatedRegionWidth + 2;
		int extNy = m_simulatedRegionHeight + 2;
		cudaCheck(cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint)));

		int x = (m_simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		synchronCheck;

		C_MoveSimulatedRegion << < blocksPerGrid, threadsPerBlock >> > (
			m_device_grid_next,
			m_simulatedRegionWidth,
			m_simulatedRegionHeight,
			nx,
			ny,
			m_grid_pitch,
			m_horizon);
		swapDeviceGrid();
		synchronCheck;

		addSource();

		m_simulatedOriginX += nx;
		m_simulatedOriginY += ny;

		//	std::cout << "Origin X: " << m_simulatedOriginX << " Origin Y: " << m_simulatedOriginY << std::endl;
	}

	__device__ float C_GetU(gridpoint gp)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;

		float h4 = h * h * h * h;
		return sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));
	}

	__device__ float C_GetV(gridpoint gp)
	{
		float h = max(gp.x, 0.0f);
		float vh = gp.z;

		float h4 = h * h * h * h;
		return sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));
	}

	__global__ void C_AddSource(
		gridpoint *grid,
		float2* source,
		int patchSize,
		int pitchSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			int gx = i + 1;
			int gy = j + 1;

			float4 gp = tex2D(g_capillaryTexture, gx, gy);
			float2 s_ij = source[i + j*patchSize];

			float h = gp.x;
			float u = C_GetU(gp);
			float v = C_GetV(gp);

			if (length(s_ij) > 0.001f)
			{
				u += s_ij.x;
				v += s_ij.y;

				u *= 0.98f;
				v *= 0.98f;

				u = min(0.4f, max(-0.4f, u));
				v = min(0.4f, max(-0.4f, v));
			}

			gp.x = h;
			gp.y = u*h;
			gp.z = v*h;

			grid2Dwrite(grid, gx, gy, pitchSize, gp);
		}
	}

	void CapillaryWave::addSource()
	{
		int x = (m_simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, m_simulatedRegionWidth + 2, m_simulatedRegionHeight + 2, m_grid_pitch * sizeof(gridpoint));
		C_AddSource << < blocksPerGrid, threadsPerBlock >> > (
			m_device_grid_next,
			m_source,
			m_simulatedRegionWidth,
			m_grid_pitch);
		swapDeviceGrid();
		synchronCheck;
	}

	__global__ void C_ImposeBC(float4* grid_next, float4* grid, int width, int height, int pitch)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < width && y < height)
		{
			if (x == 0)
			{
				float4 a = grid2Dread(grid, 1, y, pitch);
				grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else if (x == width - 1)
			{
				float4 a = grid2Dread(grid, width - 2, y, pitch);
				grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else if (y == 0)
			{
				float4 a = grid2Dread(grid, x, 1, pitch);
				grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else if (y == height - 1)
			{
				float4 a = grid2Dread(grid, x, height - 2, pitch);
				grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else
			{
				float4 a = grid2Dread(grid, x, y, pitch);
				grid2Dwrite(grid_next, x, y, pitch, a);
			}
		}
	}

	__host__ __device__ void C_FixShore(gridpoint& l, gridpoint& c, gridpoint& r)
	{

		if (r.x < 0.0f || l.x < 0.0f || c.x < 0.0f)
		{
			c.x = c.x + l.x + r.x;
			c.x = max(0.0f, c.x);
			l.x = 0.0f;
			r.x = 0.0f;
		}
		float h = c.x;
		float h4 = h * h * h * h;
		float v = sqrtf(2.0f) * h * c.y / (sqrtf(h4 + max(h4, EPSILON)));
		float u = sqrtf(2.0f) * h * c.z / (sqrtf(h4 + max(h4, EPSILON)));

		c.y = u * h;
		c.z = v * h;
	}

	__host__ __device__ gridpoint C_VerticalPotential(gridpoint gp)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float v = sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));

		gridpoint G;
		G.x = v * h;
		G.y = uh * v;
		G.z = vh * v + GRAVITY * h * h;
		G.w = 0.0f;
		return G;
	}

	__device__ gridpoint C_HorizontalPotential(gridpoint gp)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float u = sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));

		gridpoint F;
		F.x = u * h;
		F.y = uh * u + GRAVITY * h * h;
		F.z = vh * u;
		F.w = 0.0f;
		return F;
	}

	__device__ gridpoint C_SlopeForce(gridpoint c, gridpoint n, gridpoint e, gridpoint s, gridpoint w)
	{
		float h = max(c.x, 0.0f);

		gridpoint H;
		H.x = 0.0f;
		H.y = -GRAVITY * h * (e.w - w.w);
		H.z = -GRAVITY * h * (s.w - n.w);
		H.w = 0.0f;
		return H;
	}

	__global__ void C_OneWaveStep(gridpoint* grid_next, int width, int height, float timestep, int pitch)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < width && y < height)
		{
			int gridx = x + 1;
			int gridy = y + 1;

			gridpoint center = tex2D(g_capillaryTexture, gridx, gridy);

			gridpoint north = tex2D(g_capillaryTexture, gridx, gridy - 1);

			gridpoint west = tex2D(g_capillaryTexture, gridx - 1, gridy);

			gridpoint south = tex2D(g_capillaryTexture, gridx, gridy + 1);

			gridpoint east = tex2D(g_capillaryTexture, gridx + 1, gridy);

			C_FixShore(west, center, east);
			C_FixShore(north, center, south);

			gridpoint u_south = 0.5f * (south + center) - timestep * (C_VerticalPotential(south) - C_VerticalPotential(center));
			gridpoint u_north = 0.5f * (north + center) - timestep * (C_VerticalPotential(center) - C_VerticalPotential(north));
			gridpoint u_west = 0.5f * (west + center) - timestep * (C_HorizontalPotential(center) - C_HorizontalPotential(west));
			gridpoint u_east = 0.5f * (east + center) - timestep * (C_HorizontalPotential(east) - C_HorizontalPotential(center));

			gridpoint u_center = center + timestep * C_SlopeForce(center, north, east, south, west) - timestep * (C_HorizontalPotential(u_east) - C_HorizontalPotential(u_west)) - timestep * (C_VerticalPotential(u_south) - C_VerticalPotential(u_north));
			u_center.x = max(0.0f, u_center.x);

			grid2Dwrite(grid_next, gridx, gridy, pitch, u_center);
		}
	}



	__global__ void C_InitHeightField(
		float4* height,
		int patchSize,
		float horizon,
		float realSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			int gridx = i + 1;
			int gridy = j + 1;

			gridpoint gp = tex2D(g_capillaryTexture, gridx, gridy);
			height[i + j*patchSize].x = gp.x - horizon;

			float d = sqrtf((i - patchSize / 2)*(i - patchSize / 2) + (j - patchSize / 2)*(j - patchSize / 2));
			float q = d / (0.49f*patchSize);

			float weight = q < 1.0f ? 1.0f - q * q : 0.0f;
			height[i + j*patchSize].y = 1.3f * realSize * sinf(3.0f*weight*height[i + j * patchSize].x*0.5f*M_PI);

			// x component stores the original height, y component stores the normalized height, z component stores the X gradient, w component stores the Z gradient;
		}
	}

	__global__ void C_InitHeightGrad(
		float4* height,
		int patchSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			int i_minus_one = (i - 1 + patchSize) % patchSize;
			int i_plus_one = (i + 1) % patchSize;
			int j_minus_one = (j - 1 + patchSize) % patchSize;
			int j_plus_one = (j + 1) % patchSize;

			float4 Dx = (height[i_plus_one + j * patchSize] - height[i_minus_one + j * patchSize]) / 2;
			float4 Dz = (height[i + j_plus_one * patchSize] - height[i + j_minus_one * patchSize]) / 2;

			height[i + patchSize * j].z = Dx.y;
			height[i + patchSize * j].w = Dz.y;
		}
	}


	void CapillaryWave::animate(float dt)
	{
		int extNx = m_simulatedRegionWidth + 2;
		int extNy = m_simulatedRegionHeight + 2;

		cudaError_t error;
		// make dimension
		int x = (m_simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		int x1 = (extNx + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y1 = (extNy + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock1(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid1(x1, y1);

		int nStep = 1;
		float timestep = dt / nStep;


		for (int iter = 0; iter < nStep; iter++)
		{
			cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint));
			C_ImposeBC << < blocksPerGrid1, threadsPerBlock1 >> > (m_device_grid_next, m_device_grid, extNx, extNy, m_grid_pitch);
			swapDeviceGrid();
			synchronCheck;

			cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint));
			C_OneWaveStep << < blocksPerGrid, threadsPerBlock >> > (
				m_device_grid_next,
				m_simulatedRegionWidth,
				m_simulatedRegionHeight,
				1.0f*timestep,
				m_grid_pitch);
			swapDeviceGrid();
			synchronCheck;
		}

		error = cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint));
		C_InitHeightField << < blocksPerGrid, threadsPerBlock >> > (m_height, m_simulatedRegionWidth, m_horizon, m_realGridSize);
		synchronCheck;
		C_InitHeightGrad << < blocksPerGrid, threadsPerBlock >> > (m_height, m_simulatedRegionWidth);
		synchronCheck;

		cudaCheck(cudaGraphicsMapResources(1, &m_cuda_texture))
			cudaArray* cuda_height_array = nullptr;
		cudaCheck(cudaGraphicsSubResourceGetMappedArray(&cuda_height_array, m_cuda_texture, 0, 0));
		cudaCheck(cudaMemcpyToArray(cuda_height_array, 0, 0, m_height, m_simulatedRegionWidth*m_simulatedRegionHeight * sizeof(float4), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaGraphicsUnmapResources(1, &m_cuda_texture));
	}

	void CapillaryWave::resetSource()
	{
		cudaMemset(m_source, 0, m_simulatedRegionWidth* m_simulatedRegionHeight * sizeof(float2));
		cudaMemset(m_weight, 0, m_simulatedRegionWidth* m_simulatedRegionHeight * sizeof(float));
	}

	void CapillaryWave::swapDeviceGrid()
	{
		gridpoint *grid_helper = m_device_grid;
		m_device_grid = m_device_grid_next;
		m_device_grid_next = grid_helper;
	}
}
