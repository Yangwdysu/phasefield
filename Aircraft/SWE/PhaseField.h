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

namespace WetBrush {

	using namespace std;


#define RHO1 1000.0f
#define RHO2 100.0f
#define g  -9.81f;
#define samplingDistance 0.1f
	class PhaseField
	{

	public:
		PhaseField();
		~PhaseField();
		void initialize();
		void initRegion();
		void AllocateMemoery(int nx, int ny, int nz);
		void PF_solver(float dt);
		void PF_setScalarFieldBoundary(Grid1f m_cuda_phasefield, bool postive);
		void NS_solver(float substep, float t);
		void animate(float dt, gridpoint* m_mdevice_grid, size_t pitch, float4* displacement);
		void display();
		void moveDynamicRegion(int dx, int dy, glm::vec3 v);


		void setOriginX(int x) { m_simulatedOriginX = x; }
		void setOriginY(int y) { m_simulatedOriginY = y; }

		int m_simulatedOriginX = 0;			//动态区域初始x坐标
		int m_simulatedOriginY = 0;			//动态区域初始y坐标

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
		Grid3f m_cuda_Norm;
		Grid1f m_cuda_phasefieldcp;
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

		float diffusion;
		int p_simulatedOriginX = 0;
		int p_simulatedOriginY = 0;

		int nx;
		int ny;
		int nz;
		int dSize;

	};
}


