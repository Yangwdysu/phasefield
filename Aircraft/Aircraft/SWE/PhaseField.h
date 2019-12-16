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
#include <cuda_runtime.h>
using namespace std;


#define RHO1 100.0f
#define RHO2 1000.0f
#define g  -9.81f;
class PhaseField
{
	
public:
	PhaseField(int size, float patchLength);
	~PhaseField();
	void initialize();
	void initDynamicRegion();
	void AllocateMemoery(int nx, int ny, int nz);
	void PF_solver(float* phaseField, float* phaseField0, float* Velu_x, float* Velv_y, float* Velw_z, float dt);
	void PF_setScalarFieldBoundary(float* Device_field, bool postive);
	void NS_solver(float* vel_u, float* vel_v, float* vel_w, float* H_mass, float substep);
	void animate(float dt);
	float CFL();
	void display();
	void moveSimulationRegion(int nx, int ny);
	float4* getHeightField() {return m_cuda_position;}
	void updatePosition(float dt);



	float getRealGridSize() { return m_realGridSize; }
	void setOriginX(int x) { m_simulatedOriginX = x; }
	void setOriginY(int y) { m_simulatedOriginY = y; }
	int getGridSize() { return simulatedRegionLenght; }
	int getOriginX() { return m_simulatedOriginX; }
	int getOriginZ() { return m_simulatedOriginY; }


public:
           
	static float timeStep;
	//static float samplingDistance;
	static float rhoLiquidRef;
	static float rhoAirRef;
	float massAir;
	float massLiquid;
	float smoothinglength;
	float diff;

	float m_patch_length;
	float m_realGridSize;			//网格实际距离


	float3* m_cuda_Veluc1;

	float4* m_cuda_SimulationRegion;     //仿真区域位置
	rgb* m_cuda_SimulationRegionColor;

	float4* m_cuda_position;		//水体位置
	rgb* m_cuda_color;


	float* m_cuda_phasefield0;
	float* m_cuda_phasefield;


	float* m_cuda_Velu;
	float* m_cuda_Velv;
	float* m_cuda_Velw;
	float3* m_cuda_Veluc;
	float3* m_cuda_Veluc0;

	float7* m_cuda_CoefMatrix;
	float* m_cuda_RHS;
	float* m_cuda_Pressure;
	float* m_cuda_BufPressure;

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

	//计算区域/流体部分
	int nx;
	int ny;
	int nz;
	int m_fft_size;


	//仿真区域的初始位置
	int m_simulatedOriginX = 0;			//动态区域初始x坐标
	int m_simulatedOriginY = 0;			//动态区域初始y坐标
	float m_horizon = 2.0f;			//水面初始高度
	//int m_simulatedRegionWidth;		//动态区域宽度
	//int m_simulatedRegionHeight;	//动态区域高度

	//仿真区域的大小
	int simulatedRegionLenght;
	int simulatedRegionWidth;
	int simulatedRegionHeight;

	int simulationRegitionSize;
	int simulationSize;

};


