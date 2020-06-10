#pragma once
#include "types.h"
#include <cuda_runtime.h>
#include "OceanPatch.h"
#include"PhaseField.h"
namespace WetBrush {
	class CapillaryWave
	{
	public:
		CapillaryWave(int size, float patchLength);
		~CapillaryWave();

		void initialize();

		void initDynamicRegion();
		void initSource();

		void moveDynamicRegion(int nx, int ny);		//���洬���ƶ���̬��������
		//void setVelocity(gridpoint vel) { m_device_grid = vel; }
		//void getVelocity() { return m_device_grid; }
		void addSource();

		void animate(float dt, Grid3f position, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w);

		float4* getHeightField() {
			return m_height;
		}

		GLuint getHeightTextureId() { return m_height_texture; }

		float2* getSource() { return m_source; }
		float* getWeight() { return m_weight; }

		int getGridSize() { return m_simulatedRegionWidth; }

		int getPitchSize() { return m_grid_pitch; }
		float getHorizon() { return m_horizon; }
		int getOriginX() { return m_simulatedOriginX; }
		int getOriginZ() { return m_simulatedOriginY; }

		void setOriginX(int x) { m_simulatedOriginX = x; }
		void setOriginY(int y) { m_simulatedOriginY = y; }

		float2 getOrigin() { return make_float2(m_simulatedOriginX*m_realGridSize, m_simulatedOriginY*m_realGridSize); }

		float getRealGridSize() { return m_realGridSize; }

		void resetSource();

	public:
		void swapDeviceGrid();

		float m_horizon = 2.0f;			//ˮ���ʼ�߶�

		float m_patch_length;
		float m_realGridSize;			//����ʵ�ʾ���

		int m_simulatedRegionWidth;		//��̬������
		int m_simulatedRegionHeight;	//��̬����߶�

		int m_Nx;
		int m_Ny;

		int m_simulatedOriginX = 0;			//��̬�����ʼx����
		int m_simulatedOriginY = 0;			//��̬�����ʼy����

		gridpoint* m_device_grid;		//��ǰ��̬����״̬
		gridpoint* m_device_grid_next;
		size_t m_grid_pitch;

		float4* m_height = nullptr;				//�߶ȳ�
		GLuint m_height_texture = 0;		//�߶ȳ�texture
		cudaGraphicsResource_t m_cuda_texture;// cuda opengl object

		float2* m_source;				//������Ӵ���ˮ����
		float* m_weight;
	};
}

