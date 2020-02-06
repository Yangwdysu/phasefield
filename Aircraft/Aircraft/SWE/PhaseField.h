#pragma once
#include<iostream>
#include <sstream>
#include<algorithm>
#include "types.h"
#include <cuda_runtime.h>
#include "OceanPatch.h"
//#include<stdlib.h>
//#include<string.h>
//#include <stdio.h>
#include<PFDatau.h>


namespace WetBrush {

	using namespace std;


#define RHO1 1000.0f
#define RHO2 100.0f
#define g  -9.81f;
	class PhaseField
	{

	public:
		PhaseField();
		~PhaseField();
		void initialize();
		void initRegion();
		void moveSimulationRegion(int nx, int ny);
		void AllocateMemoery(int nx, int ny, int nz);
		void PF_solver(float dt);
		void PF_setScalarFieldBoundary( bool postive);
		void NS_solver(float substep);
		void animate(float dt);
		float CFL(float max_velu);
		void display();
		void updatePosition(float dt);



		float getRealGridSize() { return m_realGridSize; }
		void setOriginX(int x) { m_simulatedOriginX = x; }
		void setOriginY(int y) { m_simulatedOriginY = y; }
		int getGridSize() { return simulatedRegionLenght; }
		int getOriginX() { return m_simulatedOriginX; }
		int getOriginZ() { return m_simulatedOriginY; }
		//float4* getHeightField() { return m_cuda_position; }

	public:

		static float timeStep;
		//static float samplingDistance;
		static float rhoLiquidRef;
		static float rhoAirRef;
		float massAir;
		float massLiquid;
		float smoothinglength;
		float diff;






		float4* m_cuda_SimulationRegion;     //��������λ��
		rgb* m_cuda_SimulationRegionColor;

		Grid3f m_cuda_position;		//ˮ��λ��
		rgb* m_cuda_color;


		Grid1f m_cuda_phasefield0;
		Grid1f m_cuda_phasefield;
		float* max_velu;
		float* cfl;

		Grid1f m_cuda_Velu;
		Grid1f m_cuda_Velv;
		Grid1f m_cuda_Velw;


		Grid3f m_cuda_Veluc;
		Grid3f m_cuda_Veluc0;
		Grid3f m_cuda_Veluc1;

		GridCoef m_cuda_CoefMatrix;
		Grid1f m_cuda_RHS;
		Grid1f m_cuda_Pressure;
		Grid1f m_cuda_BufPressure;

		GLuint
			m_indexbufferID,
			m_watersurface[2],
			m_indexbufferID1,
			SimulationRegion_bufferObj,
			Initpos_bufferObj,
			PhaseField_bufferObj,
			SimulationRegionColor_bufferObj,
			Color_bufferObj,
			Velocity_bufferObj[3];

		struct cudaGraphicsResource
			*SimulationRegion_resource,
			*Initpos_resource,
			*PhaseField_resource,
			*SimulationRegionColor_resource,
			*Color_resource,
			*Velu_resource,
			*Velv_resource,
			*Velw_resource;




		int simItor;
		float m_patch_length;
		float m_realGridSize;			//����ʵ�ʾ���

		//��������/���岿��
		int nx;
		int ny;
		int nz;
		int sizem;


		//��������ĳ�ʼλ��
		int m_simulatedOriginX = 0;			//��̬�����ʼx����
		int m_simulatedOriginY = 0;			//��̬�����ʼy����
		float m_horizon = 2.0f;			//ˮ���ʼ�߶�
		//int m_simulatedRegionWidth;		//��̬������
		//int m_simulatedRegionHeight;	//��̬����߶�

		//��������Ĵ�С
		int simulatedRegionLenght;
		int simulatedRegionWidth;
		int simulatedRegionHeight;

		int simulationRegitionSize;
		int simulationSize;

	};
}


