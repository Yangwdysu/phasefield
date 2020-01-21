#pragma once
#include "types.h"
#include <cuda_runtime.h>
#include <math.h>
#include "OceanPatch.h"
#include "CapillaryWave.h"
namespace WetBrush {
	using namespace std;
	class Ocean
	{
	public:
		Ocean();
		~Ocean();

		void initialize();

		void initWholeRegion();

		vertex* mapOceanVertex();
		rgb* mapOceanColor();

		void unmapOceanVertex();
		void unmapOceanColor();

		void display();

		void animate(float dt);

		//	void moveDynamicRegion(int nx, int ny);
		// 	GLuint getDisplacementTextureId() { return m_patch->getDisplacementTextureId(); }
		// 	GLuint getGradientTextureId() { return m_patch->getGradientTextureId(); }

		float2 getWindDirection() { return make_float2(cosf(m_patch->windDir), sinf(m_patch->windDir)); }
		float getFftRealSize() { return m_fft_real_size; }
		int getFftResolution() { return m_fft_size; }
		float getChoppiness() { return m_choppiness; }
		void setChoppiness(float chopiness) {
			m_choppiness = chopiness;
			m_patch->setChoppiness(m_choppiness);
		}
		OceanPatch* getOceanPatch() { return m_patch; }
		int getGridSize() { return m_fft_size; }

		//返回每个块实际覆盖的距离
		float getPatchLength();
		float getGridLength();

		void addTrail(CapillaryWave* trail);

		CapillaryWave* getTrail(int i) { if (i < m_trails.size())return m_trails[i]; else return nullptr; }

		void addOceanTrails(vertex* oceanVertex);

	public:
		//挤压波形形成尖浪
		float m_choppiness = 1.0f;

		//初始风级
		int m_windType = 3;

		//fft纹理分辨率
		int m_fft_size = 512;

		//fft纹理贴图的物理长度 单位米
		float m_fft_real_size = 2.0f;

		//高度场贴图的物理长度 单位米
		float m_heightfield_real_size = 5000;

		GLuint m_indexbufferID;

		GLuint m_watersurface[2];

		int m_oceanWidth;
		int m_oceanHeight;

		float m_eclipsedTime;

		float m_patchSize;
		float m_virtualGridSize;
		float m_realGridSize;

		int Nx = 4;
		int Ny = 4;

		cudaGraphicsResource* m_cudaColor;
		cudaGraphicsResource* m_cudaVertex;

	private:

		//fft wave simulation object
		OceanPatch* m_patch = nullptr;

		std::vector<CapillaryWave*> m_trails;
	};
}

