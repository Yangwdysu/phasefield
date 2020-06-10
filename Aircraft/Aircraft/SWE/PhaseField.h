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
#include <glm/glm.hpp>
#include "RigidBody\RigidBody.h"

#include"CapillaryWave.h"
namespace WetBrush {

	using namespace std;


#define RHO1 1000.0f
#define RHO2 100.0f
#define g  -9.81f;
#define samplingDistance 0.005f
	class PhaseField
	{

	public:
		PhaseField();
		~PhaseField();
		void initialize();
		void initRegion();
		void moveDynamicRegion(int nx, int ny, glm::vec3 v);

		void AllocateMemoery(int nx, int ny, int nz);
		void PF_solver(float dt);
		void PF_setScalarFieldBoundary( bool postive);
		void NS_solver(float substep, float t);
		void animate(float dt);
		void display();


		//for test
		void setOriginX(int x) { p_simulatedOriginX = x; }
		void setOriginY(int y) { p_simulatedOriginY = y; }
		int getGridSize() { return p_simulatedRegionWidth; }
		float getRealGridSize() { return p_realGridSize; }



	public:

		static float timeStep;
		//static float samplingDistance;
		static float rhoLiquidRef;
		static float rhoAirRef;
		float massAir;
		float massLiquid;
		float smoothinglength;
		float diff;


		float4* m_cuda_SimulationRegion;     //仿真区域位置
		rgb* m_cuda_SimulationRegionColor;
		Grid3f m_cuda_position;		//水体位置
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
			*Color_resource;


		CapillaryWave* mm_trail;


		int p_simulatedRegionHeight;
		int p_simulatedRegionWidth;


		int p_simulatedOriginX = 0;			//动态区域初始x坐标
		int p_simulatedOriginY = 0;			//动态区域初始y坐标
		int p_realGridSize;


		int nx;
		int ny;
		int nz;
		int dSize;

	};
}


