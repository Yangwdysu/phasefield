#if __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/glew.h>
#endif

#include <iostream>
#include "Ocean.h"
#include "cuda_helper_math.h"
#include <cuda_gl_interop.h>  
#include <cufft.h>

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

Ocean::Ocean()
{
	m_eclipsedTime = 0.0f;

	m_virtualGridSize = 0.1f;

	m_patchSize = 512.0f;
	m_oceanWidth = m_fft_size*Nx;
	m_oceanHeight = m_fft_size*Ny;

	m_choppiness = 1.0f;

	m_patch = new OceanPatch(m_fft_size, m_patchSize, m_windType);

	m_realGridSize = m_patch->getGridLength();

}

Ocean::~Ocean()
{
	delete m_patch;
}

void Ocean::initialize()
{
	initWholeRegion();
}

void Ocean::initWholeRegion()
{
	int *indices = (int *)malloc(4 * (m_oceanWidth - 1) * (m_oceanHeight - 1) * sizeof(int));

	for (int y = 0; y < m_oceanWidth - 1; y++)
	{
		for (int x = 0; x < m_oceanHeight - 1; x++)
		{
			indices[4 * (y * (m_oceanWidth - 1) + x) + 0] = (y + 1) * m_oceanWidth + x;
			indices[4 * (y * (m_oceanWidth - 1) + x) + 1] = (y + 1) * m_oceanWidth + x + 1;
			indices[4 * (y * (m_oceanWidth - 1) + x) + 2] = y * m_oceanWidth + x + 1;
			indices[4 * (y * (m_oceanWidth - 1) + x) + 3] = y * m_oceanWidth + x;
		}
	}

	vertex* wave = new vertex[m_oceanWidth*m_oceanHeight];
	rgb* color = new rgb[m_oceanWidth*m_oceanHeight];

	for (int j = 0; j < m_oceanHeight; j++)
	{
		for (int i = 0; i < m_oceanWidth; i++)
		{
			vertex vij;
			vij.x = (float)i*m_virtualGridSize;
			vij.y = (float)0;
			vij.z = (float)j*m_virtualGridSize;
			vij.w = (float)1;
			wave[j*m_oceanWidth +i] = vij;
			color[j*m_oceanWidth + i] = make_uchar4(0, 0, 255, 220);
		}
	}

	glGenBuffers(2, &m_indexbufferID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexbufferID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * (m_oceanWidth - 1) * (m_oceanWidth - 1) * sizeof(int), indices, GL_STATIC_DRAW);

	glGenBuffers(2, &m_watersurface[0]);
	glBindBuffer(GL_ARRAY_BUFFER, m_watersurface[1]);
	glBufferData(GL_ARRAY_BUFFER, m_oceanWidth * m_oceanHeight * sizeof(rgb), color, GL_STATIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&m_cudaColor, m_watersurface[1], cudaGraphicsMapFlagsWriteDiscard);

	glBindBuffer(GL_ARRAY_BUFFER, m_watersurface[0]);
	glBufferData(GL_ARRAY_BUFFER, m_oceanWidth * m_oceanHeight * sizeof(vertex), wave, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&m_cudaVertex, m_watersurface[0], cudaGraphicsMapFlagsWriteDiscard);

	delete[] indices;
	delete[] wave;
	delete[] color;
}

vertex* Ocean::mapOceanVertex()
{
	vertex* vb;
	size_t numBytes;
	cudaGraphicsMapResources(1, &m_cudaVertex, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&vb, &numBytes, m_cudaVertex);

	return vb;
}

rgb* Ocean::mapOceanColor()
{
	rgb* vc;
	size_t numBytes;
	cudaGraphicsMapResources(1, &m_cudaColor, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&vc, &numBytes, m_cudaColor);

	return vc;
}

void Ocean::unmapOceanVertex()
{
	cudaGraphicsUnmapResources(1, &m_cudaVertex, 0);
}

void Ocean::unmapOceanColor()
{
	cudaGraphicsUnmapResources(1, &m_cudaColor, 0);
}

__global__ void O_InitOceanWave(
	vertex* oceanVertex,
	rgb* oceanColor,
	float4* displacement,
	int oceanWidth,
	int oceanHeight,
	int width,
	int height,
	float choppiness,
	float realSize,
	float virtualSize,
	float globalShift,
	int wintType)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < width && j < height)
	{
		int tiledX = oceanWidth / width;
		int tiledY = oceanHeight / height;

		const float largeScale = 0.001f;

		int id = i + j*width;
		float4 D_ij = displacement[id];

		float scaledSize = realSize *virtualSize;
		vertex v;
		for (int t = 0; t < tiledX; t++)
		{
			for (int s = 0; s < tiledY; s++)
			{
				int nx = i + t*width;
				int ny = j + s*height;

				float tx = nx * largeScale;
				float ty = ny * largeScale;

				float fx = tx - floor(tx);
				float fy = ty - floor(ty);
				fx *= (width - 1);
				fy *= (width - 1);
				int lx = floor(fx);
				int ly = floor(fy);

				fx -= lx;
				fy -= ly;

				int id = lx + ly*width;
				float4 d00 = displacement[id];
				float4 d10 = displacement[id+1];
				float4 d01 = displacement[id + width];
				float4 d11 = displacement[id + width + 1];

				float4 df = d00*(1 - fx)*(1 - fy) + d10*fx*(1 - fy) + d01*(1 - fx)*fy + d11*fx*fy;

				v.x = nx * scaledSize + choppiness* scaledSize * D_ij.x +df.x * scaledSize * globalShift;
				v.y = D_ij.y*scaledSize +df.y * scaledSize * globalShift;
				v.z = ny * scaledSize + choppiness*scaledSize*D_ij.z +df.y * scaledSize * globalShift;

				rgb c;
				float ws = wintType < 1 ? 1 : (float)wintType;
				float h_ij = 0.2f*D_ij.y/ wintType;
				c.x = max(0, min(20 + (h_ij) * 500.0f, 255));
				c.y = max(0, min(40 + (h_ij) * 500.0f, 255));
				c.z = max(0, min(100 + (h_ij) * 500.0f, 255));
				c.w = 255 - max(-50 * h_ij + 50, 0);

				oceanColor[ny * oceanWidth + nx] = c;
				oceanVertex[ny * oceanWidth + nx] = v;
			}
		}
	}
}

__global__ void O_Visualise(
	vertex* oceanVertex,
	rgb* oceanColor,
	int oceanWidth,
	int oceanHeight,
	int patchSize,
	float realSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < patchSize && j < patchSize)
	{
		int tiledX = oceanWidth / patchSize;
		int tiledY = oceanHeight / patchSize;
		
		for (int t = 0; t < tiledX; t++)
		{
			for (int s = 0; s < tiledY; s++)
			{
				vertex v = oceanVertex[(j + s*patchSize) * oceanWidth + (i + t*patchSize)];
				
				rgb c;
				float h_ij = v.y / realSize;
// 				if (i + t*patchSize > 1 && i + t*patchSize < oceanWidth - 2 && j + s*patchSize < oceanHeight - 2 && j + s*patchSize > 1)
// 				{
// 					vertex v_plus_i = oceanVertex[(j + s*patchSize) * oceanWidth + (i + 1 + t*patchSize)];
// 					vertex v_minus_i = oceanVertex[(j + s*patchSize) * oceanWidth + (i - 1 + t*patchSize)];
// 					vertex v_plus_j = oceanVertex[(j + 1 + s*patchSize) * oceanWidth + (i + t*patchSize)];
// 					vertex v_minus_j = oceanVertex[(j - 1 + s*patchSize) * oceanWidth + (i + t*patchSize)];
// 					h_ij = (v_plus_i.y + v_minus_i.y + v_minus_j.y + v_plus_j.y - 4.0f*v.y);
// 				}

				c.x = max(0, min(20 + (h_ij) * 500.0f, 255));
				c.y = max(0, min(40 + (h_ij) * 500.0f, 255));
				c.z = max(0, min(100 + (h_ij) * 500.0f, 255));
				c.w = 255 - max(-50 * h_ij + 50, 0);

				oceanColor[(j + s*patchSize) * oceanWidth + (i + t*patchSize)] = c;
			}
		}
	}
}

__global__ void O_AddCapillaryWave(
	vertex* oceanVertex,
	float4* heightfield,
	int waveGridSize,
	int oceanWidth,
	int oceanHeight,
	int originX,
	int originY,
	float waveSpacing,
	float oceanSpacing,
	float horizon,
	float realSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (i < waveGridSize && j < waveGridSize)
	{
		float d = sqrtf((i-waveGridSize/2)*(i - waveGridSize / 2) + (j - waveGridSize / 2)*(j - waveGridSize / 2));
		float q = d / (0.49f*waveGridSize);

		float weight = q < 1.0f ? 1.0f - q*q : 0.0f;

		int oi = (i + originX) * waveSpacing / oceanSpacing;
		int oj = (j + originY) * waveSpacing / oceanSpacing;

		if (oi > 0 && oi < oceanWidth && oj > 0 && oj < oceanHeight)
		{
			int ocean_id = oj * oceanWidth + oi;
			int hf_id = j*waveGridSize + i;
			float h_ij = heightfield[hf_id].x;
			vertex o_ij = oceanVertex[ocean_id];

			float value = sin(3.0f*weight*h_ij*0.5f*M_PI);
			o_ij.y += realSize*value;// 3.0f*weight*realSize*h_ij;

			oceanVertex[ocean_id] = o_ij;
		}
	}
}

void Ocean::display()
{
	glEnableClientState(GL_INDEX_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexbufferID);
	glIndexPointer(GL_INT, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_watersurface[1]);
	glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(rgb), 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_watersurface[0]);
	glVertexPointer(3, GL_FLOAT, sizeof(vertex), 0);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDrawElements(GL_QUADS, 4 * (m_oceanWidth - 1) * (m_oceanHeight - 1), GL_UNSIGNED_INT, 0);
	glDisable(GL_BLEND);

//	glDrawElements(GL_POINTS, 4 * (m_oceanWidth - 1) * (m_oceanHeight - 1), GL_UNSIGNED_INT, 0);

	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_INDEX_ARRAY);
}

void Ocean::animate(float dt)
{
	m_patch->animate(m_eclipsedTime);
	m_eclipsedTime += dt;

	cudaError_t error;
	// make dimension
	int x = (m_fft_size + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
	int y = (m_fft_size + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
	dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
	dim3 blocksPerGrid(x, y);

	vertex* oceanVertex = mapOceanVertex();
	rgb* oceanColor = mapOceanColor();

	O_InitOceanWave << < blocksPerGrid, threadsPerBlock >> > (
		oceanVertex,
		oceanColor,
		m_patch->getDisplacement(),
		m_oceanWidth,
		m_oceanHeight,
		m_fft_size,
		m_fft_size,
		m_patch->getChoppiness()*m_patch->getMaxChoppiness(),
		m_realGridSize,
		m_virtualGridSize,
		0.0f*m_patch->getGlobalShift(),
		m_windType);

	addOceanTrails(oceanVertex);

// 	O_AddCapillaryWave << < blocksPerGrid, threadsPerBlock >> > (
// 		oceanVertex,
// 		m_capillay->getHeightField(),
// 		m_capillay->getGridSize(),
// 		m_oceanWidth,
// 		m_oceanHeight,
// 		m_simulatedOriginX,
// 		m_simulatedOriginY,
// 		m_capillay->getHorizon(),
// 		m_realGridSize);

	O_Visualise << <blocksPerGrid, threadsPerBlock >> > (
		oceanVertex,
		oceanColor,
		m_oceanWidth,
		m_oceanHeight,
		m_fft_size,
		m_realGridSize);

	unmapOceanVertex();
	unmapOceanColor();
}

float Ocean::getPatchLength()
{
	return m_patchSize;
}

float Ocean::getGridLength()
{
	return m_patchSize / m_fft_size;
}

// void Ocean::moveDynamicRegion(int nx, int ny)
// {
// 	int new_x = m_simulatedOriginX + nx;
// 	int new_y = m_simulatedOriginY + ny;
// 
// 	new_x = max(0, min(new_x, m_oceanWidth - m_heightfield_size));
// 	new_y = max(0, min(new_y, m_oceanHeight - m_heightfield_size));
// 
// 	m_simulatedOriginX = new_x;
// 	m_simulatedOriginY = new_y;
// }

void Ocean::addTrail(CapillaryWave* trail)
{
	m_trails.push_back(trail);
}

void Ocean::addOceanTrails(vertex* oceanVertex)
{
	int x = (m_fft_size + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
	int y = (m_fft_size + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
	dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
	dim3 blocksPerGrid(x, y);

	for (size_t i = 0; i < m_trails.size(); i++)
	{
		auto trail = m_trails[i];
		O_AddCapillaryWave << < blocksPerGrid, threadsPerBlock >> > (
			oceanVertex,
			trail->getHeightField(),
			trail->getGridSize(),
			m_oceanWidth,
			m_oceanHeight,
			trail->getOriginX(),
			trail->getOriginZ(),
			trail->getRealGridSize(),
			getGridLength(),
			trail->getHorizon(),
			0.5f);
	}
}
