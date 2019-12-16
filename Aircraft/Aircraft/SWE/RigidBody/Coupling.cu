#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#include "Coupling.h"
#include "RigidBody.h"
#include <cuda_runtime.h>
#include "../cuda_helper_math.h"
#include "cuda_utilities.h"
#include <math.h>
#include <iostream>
#include "stdio.h"

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define mf 0.000001

static void printmat4(glm::dmat4 mat)
{
	printf("mat4x4\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n"
		, mat[0][0], mat[1][0], mat[2][0], mat[3][0]
		, mat[0][1], mat[1][1], mat[2][1], mat[3][1]
		, mat[0][2], mat[1][2], mat[2][2], mat[3][2]
		, mat[0][3], mat[1][3], mat[2][3], mat[3][3]);
}

static void printmat3(glm::mat3 mat)
{
	printf("mat3x3\n%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n"
		, mat[0][0], mat[0][1], mat[0][2]
		, mat[1][0], mat[1][1], mat[1][2]
		, mat[2][0], mat[2][1], mat[2][2]);
}

Coupling::Coupling(OceanPatch* ocean_patch)
{
	m_ocean_patch = ocean_patch;
	m_origin = make_float2(0.0f);

	m_trail = NULL;
	m_boat = NULL;

	m_force_corrected = false;

	m_heightShift = 6.0f;
	m_eclipsedTime = 0.0f;

	m_prePos = glm::vec3(0);
}

Coupling::~Coupling()
{
	cudaFree(m_forceX);
	cudaFree(m_forceY);
	cudaFree(m_forceZ);
	cudaFree(m_torqueX);
	cudaFree(m_torqueY);
	cudaFree(m_torqueZ);
	cudaFree(m_sample_heights);
//	cudaFree(m_oceanCentroid);
}

__device__ float4 getDisplacement(float3 pos, float4* oceanPatch, float2 origin, float patchSize, int gridSize)
{
	float2 uv_i = (make_float2(pos.x, pos.z) - origin) / patchSize;
	float u = (uv_i.x - floor(uv_i.x))*gridSize;
	float v = (uv_i.y - floor(uv_i.y))*gridSize;
	int i = floor(u);
	int j = floor(v);
	float fx = u - i;
	float fy = v - j;
	if (i == gridSize - 1)
	{
		i = gridSize - 2;
		fx = 1.0f;
	}
	if (j == gridSize - 1)
	{
		j = gridSize - 2;
		fy = 1.0f;
	}
	int id = i + j*gridSize;
	float4 d00 = oceanPatch[id];
	float4 d10 = oceanPatch[id + 1];
	float4 d01 = oceanPatch[id + gridSize];
	float4 d11 = oceanPatch[id + gridSize + 1];

	return d00*(1 - fx)*(1 - fy) + d10*fx*(1 - fy) + d01*(1 - fx)*fy + d11*fx*fy;
}


__global__ void C_ComputeForceAndTorque(
	float* forceX,
	float* forceY,
	float* forceZ,
	float* torqueX,
	float* torqueY,
	float* torqueZ,
	float* sampleHeights,
	float3* normals,
	float3* samples,
	float4* ocean,
	glm::vec3 boatCenter,
	glm::mat3 rotation,
	float2 origin,
	int numOfSamples,
	float patchSize,
	int gridSize)
{
	int pId = threadIdx.x + blockIdx.x * blockDim.x;
	if (pId < numOfSamples)
	{
		float3 dir_i = samples[pId];
		glm::vec3 rotDir = rotation*glm::vec3(dir_i.x, dir_i.y, dir_i.z);
		glm::vec3 pos_i = boatCenter + rotation*glm::vec3(dir_i.x, dir_i.y, dir_i.z);
		float4 dis_i = getDisplacement(make_float3(pos_i.x, pos_i.y, pos_i.z), ocean, origin, patchSize, gridSize);
		dis_i.y *= 1.0f;
		float3 normal_i = normals[pId];
		float3 force_i = make_float3(0.0f);
		float3 torque_i = make_float3(0.0f);

		if (pos_i.y < dis_i.y)
		{
			force_i = 9800.0f*normal_i*(dis_i.y - pos_i.y);
			torque_i =  make_float3(0.0f, 9800.0f, 0.0f)*(dis_i.y - pos_i.y);
		}
		
		torque_i = cross(make_float3(rotDir.x, rotDir.y, rotDir.z), torque_i);
		forceX[pId] = force_i.x;
		forceY[pId] = force_i.y;
		forceZ[pId] = force_i.z;
		torqueX[pId] = torque_i.x;
		torqueY[pId] = torque_i.y;
		torqueZ[pId] = torque_i.z;
		sampleHeights[pId] = dis_i.y;
	}
	
}

__global__ void C_ComputeElevation(
	float3* displacement,
	float3 pos,
	float4* ocean,
	float2 origin,
	float patchSize,
	int gridSize)
{
	float4 tmp = getDisplacement(pos, ocean, origin, patchSize, gridSize);
	displacement[0] = make_float3(tmp.x, tmp.y, tmp.z);
}

__global__ void C_ComputeTrail(
	float2* trails,
	float* weights,
	int trail_size,
	float2 trail_origin,
	float trail_grid_distance,
	float3* samples,
	int sample_size,
	glm::vec2 boat_velocity,
	glm::vec3 boat_center,
	glm::mat3 boat_rotation,
	float t)
{
	int pId = threadIdx.x + blockIdx.x * blockDim.x;
	if (pId < sample_size)
	{
		float3 dir_i = samples[pId];
		if (abs(dir_i.z) < 120.0f && abs(dir_i.x) < 30.0f)
		{
			glm::vec3 pos_i = boat_center + boat_rotation*glm::vec3(dir_i.x, dir_i.y, dir_i.z);
			float2 local_pi = (make_float2(pos_i.x, pos_i.z) - trail_origin) / trail_grid_distance;
			int i = floor(local_pi.x);
			int j = floor(local_pi.y);

			glm::mat3 aniso(2.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.5f);
			aniso = boat_rotation*aniso*glm::transpose(boat_rotation);

			int r = 5;
			for (int s = i - r; s <= i + r; s++)
			{
				for (int t = j - r; t <= j + r; t++)
				{
					float dx = s - i;
					float dz = t - j;
					glm::vec3 rotated = aniso*glm::vec3(dx, 0.0f, dz);


					float d = sqrt(rotated.x*rotated.x + rotated.z*rotated.z);
					if (d < r)
					{
						float2 dw = (1.0f - d / r)*0.005f*make_float2(boat_velocity.x, boat_velocity.y);
						atomicAdd(&trails[s + t*trail_size].x, dw.x);
						atomicAdd(&trails[s + t*trail_size].y, dw.y);
						atomicAdd(&weights[s + t*trail_size], 1.0f);
						//trails[s + t*trail_size] = (1.0f-d/r)*0.03f*make_float2(boat_velocity.x, boat_velocity.y);
					}
				}
			}

			

			//printf("%d, %d, %f %f \n", i, j, trails[i + j * trail_size].x, trails[i + j * trail_size].y);
		}
	}
}

__global__ void C_NormalizeTrail(
	float2* trails,
	float* weights,
	int trail_size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < trail_size && j < trail_size)
	{
		int id = i + trail_size * j;
		float w = weights[id];
		if (w > 1.0f)
		{
			trails[id] /= w;
		}
	}
}

void Coupling::animate(float dt)
{
	m_eclipsedTime += dt;

	synchronCheck;
	unsigned int pDims = Physika::cudaGridSize((unsigned int)m_boat->getSamplingPointSize(), 64);

	C_ComputeForceAndTorque << <pDims, 64 >> > (
		m_forceX,
		m_forceY,
		m_forceZ,
		m_torqueX,
		m_torqueY,
		m_torqueZ,
		m_sample_heights,
		m_boat->getNormals(),
		m_boat->getSamples(),
		m_ocean_patch->getDisplacement(),
		m_boat->getCenter(),
		m_boat->getOrientation(),
		m_origin,
		m_boat->getSamplingPointSize(),
		m_ocean_patch->getPatchSize(),
		m_ocean_patch->getGridSize());
	synchronCheck;
// 	C_ComputeElevation << < 1, 1 >> > (
// 		m_oceanCentroid,
// 		make_float3(m_boat->getCenter().x, m_boat->getCenter().y, m_boat->getCenter().z),
// 		m_ocean_patch->getDisplacement(),
// 		m_origin,
// 		m_ocean_patch->getPatchSize(),
// 		m_ocean_patch->getGridSize());
// 	synchronCheck;
// 	float3 oceanCenter;
// 	cudaMemcpy(&oceanCenter, m_oceanCentroid, sizeof(float3), cudaMemcpyDeviceToHost);

	float fx = m_reduce->Accumulate(m_forceX, m_boat->getSamplingPointSize());
	float fy = m_reduce->Accumulate(m_forceY, m_boat->getSamplingPointSize());
	float fz = m_reduce->Accumulate(m_forceZ, m_boat->getSamplingPointSize());

	float tx = m_reduce->Accumulate(m_torqueX, m_boat->getSamplingPointSize());
	float ty = m_reduce->Accumulate(m_torqueY, m_boat->getSamplingPointSize());
	float tz = m_reduce->Accumulate(m_torqueZ, m_boat->getSamplingPointSize());

	float h = m_reduce->Accumulate(m_sample_heights, m_boat->getSamplingPointSize());


// 	std::cout << "Center: " << m_rigids->getCenter().x << " " << m_rigids->getCenter().y << " " << m_rigids->getCenter().z << std::endl;
// 	std::cout << "Velocity: " << m_rigids->getVelocity().x << " " << m_rigids->getVelocity().y << " " << m_rigids->getVelocity().z << std::endl;
// 	std::cout << "Angular V: " << m_rigids->getAngularVelocity().x << " " << m_rigids->getAngularVelocity().y << " " << m_rigids->getAngularVelocity().z << std::endl;
// 	std::cout << "Euler: " << m_rigids->getYaw() << " " << m_rigids->getPitch() << " " << m_rigids->getRoll() << std::endl;

	int num = m_boat->getSamplingPointSize();

	glm::vec3 force = glm::vec3(fx / num, 0.0f, fz / num);
	glm::vec3 torque = glm::vec3(tx / num, ty / num, tz / num);
	if (!m_force_corrected)
	{
		m_force_corrector = force;
		m_torque_corrector = torque;
		m_force_corrected = true;
	}

	m_boat->update(force - m_force_corrector, torque - m_torque_corrector, dt);
//	std::cout << "Angular Force: " << tx << " " << ty << " " << tz << std::endl;
	//m_boat->update(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(tx / num, ty / num, tz / num), dt);
	m_boat->advect(dt);
	synchronCheck;
	glm::vec3 center = m_boat->getCenter();
	center.y = h / m_boat->getSamplingPointSize() + m_heightShift;
	m_boat->setCenter(center);

	int originX = m_trail->getOriginX();
	int originZ = m_trail->getOriginZ();

	float dg = m_trail->getRealGridSize();
	int gridSize = m_trail->getGridSize();
	int new_x = floor(center.x / dg) - gridSize / 2;
	int new_z = floor(center.z / dg) - gridSize / 2;

// 	auto lc = getLocalBoatCenter();
// 	std::cout << "Center: " << lc.x << " " << lc.y << std::endl;

	//new_x = max(0, min(new_x, 3 * gridSize));
	//new_z = max(0, min(new_z, 3 * gridSize));
	if (abs(new_x - originX) > 20 || abs(new_z - originZ) > 20)
	{
		m_trail->setOriginX(new_x);
		m_trail->setOriginY(new_z);
		//m_phasefield->setOriginX(new_x);
		//m_phasefield->setOriginY(new_z);
	}
	else
	{
		m_trail->moveDynamicRegion(new_x - originX, new_z - originZ);
		//m_phasefield->moveSimulationRegion(new_x - originX, new_z - originZ);
	}

	/*
	date:2019/12/15
	author:@wdy
	describe: for test moveRegion
	*/
	int originX1 = m_phasefield->getOriginX();
	int originZ1 = m_phasefield->getOriginZ();
	float dg1 = m_phasefield->getRealGridSize();
	int gridSize1 = m_phasefield->getGridSize();
	int new_x1 = floor(center.x / dg1) - gridSize1 / 2;
	int new_z1 = floor(center.z / dg1) - gridSize1 / 2;
	if (abs(new_x1 - originX1) > 20 || abs(new_z1 - originZ1) > 20)
	{
		m_phasefield->setOriginX(new_x1);
		m_phasefield->setOriginY(new_z1);
	}
	else
		m_phasefield->moveSimulationRegion(new_x1 - originX1, new_z1 - originZ1);

	m_trail->animate(dt);
	m_trail->resetSource();
	//m_phasefield->animate(dt);//for pahsefield
	//synchronCheck;
	
	glm::vec3 v = m_boat->getVelocity();
	
// 	glm::vec3 an = m_boat->getAngularVelocity();
// 	std::cout << m_name << " velocity: " << v.x << " " << v.z << std::endl;
// 	std::cout << m_name << " angular velocity: " << an.x << " " << an.y << " " << an.z << std::endl;

	C_ComputeTrail << <pDims, 64 >> > (
		m_trail->getSource(),
		m_trail->getWeight(),
		m_trail->getGridSize(),
		m_trail->getOrigin(),
		m_trail->getRealGridSize(),
		m_boat->getSamples(),
		m_boat->getSamplingPointSize(),
		glm::vec2(v.x, v.z),
		m_boat->getCenter(),
		m_boat->getOrientation(),
		m_eclipsedTime);

	int x = (m_trail->getGridSize() + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
	int y = (m_trail->getGridSize() + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
	dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
	dim3 blocksPerGrid(x, y);
	C_NormalizeTrail<< < blocksPerGrid, threadsPerBlock >> > (m_trail->getSource(), m_trail->getWeight(), m_trail->getGridSize());
	synchronCheck;
}

void Coupling::setHeightShift(float shift)
{
	m_heightShift = shift;
}

void Coupling::setBoatMatrix(glm::dmat4 mat, float dt)
{
	printf("******************set boat matrix\n");
	//printmat4(mat);
	glm::mat3 rotation(
		mat[0][0], mat[0][1], mat[0][2],
		mat[1][0], mat[1][1], mat[1][2],
		mat[2][0], mat[2][1], mat[2][2]);

	//printmat3(rotation);
	//printmat4(mat);

	glm::mat3 forward;
	Quaternion<float> quat = Quaternion<float>(cos(M_PI / 4.0f), 0.0f, sin(M_PI / 4.0f), 0.0f)*Quaternion<float>(cos(M_PI / 4.0f), sin(M_PI / 4.0f), 0.0f, 0.0f);
	quat.Quaternion2Matrix(&forward[0][0]);
	//printmat3(forward);

	rotation = rotation;
	//rotation = const_rot*rotation;
	//glm::mat3 rotation(
	//	mat[0][0], mat[1][0], mat[2][0],
	//	mat[0][1], mat[1][1], mat[2][1],
	//	mat[0][2], mat[1][2], mat[2][2]);
	auto qutern = Quaternion<float>::Matrix2Quaternion(&rotation[0][0]);
	m_boat->setQuaternian(Quaternion<float>(qutern.Gets(), qutern.Getx(), qutern.Getz(), qutern.Gety()));
	//m_boat->setQuaternian(Quaternion<float>::Matrix2Quaternion(&rotation[0][0]));

	m_boat->setCenter(glm::vec3(mat[3][0], mat[3][1], mat[3][2]));

	glm::vec3 curPos = glm::vec3(mat[3][0], mat[3][1], mat[3][2]);
	if (glm::length(curPos - m_prePos) < 2.0f)
	{
		glm::vec3 vel = (curPos - m_prePos) / dt;
		std::cout << "Velocity: " << vel.x << " " << vel.y << " " << vel.z << " " << dt << std::endl;
		m_boat->setVelocity(vel);
	}
	m_prePos = curPos;
}

glm::dmat4 Coupling::getBoatMatrix()
{
	glm::mat3 backward;
	Quaternion<float> quat2 = Quaternion<float>(cos(-M_PI / 4.0f), sin(-M_PI / 4.0f), 0.0f, 0.0f)*Quaternion<float>(cos(-M_PI / 4.0f), 0.0f, sin(-M_PI / 4.0f), 0.0f);
	quat2.Quaternion2Matrix(&backward[0][0]);

	auto quat = m_boat->getQuaternian();
	quat = Quaternion<float>(quat.Gets(), quat.Getx(), quat.Getz(), quat.Gety());
	glm::mat3 rotation;
	quat.Quaternion2Matrix(&rotation[0][0]);
	auto position = m_boat->getCenter();

	//glm::mat3 corrected = backward*rotation;
	glm::mat3 corrected = rotation;

	auto result = glm::dmat4(
		corrected[0][0], corrected[0][1], corrected[0][2], 0,
		corrected[1][0], corrected[1][1], corrected[1][2], 0,
		corrected[2][0], corrected[2][1], corrected[2][2], 0,
		position[0], position[1], position[2], 1);
	//printf("----------------get boat matrix\n");
	//printmat3(corrected);
	//printmat4(result);
	return result;
}

void Coupling::steer(float degree)
{
	float yaw, pitch, roll;
	m_boat->getEulerAngle(yaw, pitch, roll);
	std::cout << "Before: yaw: " << yaw << " Pitch: " << pitch << " roll: " << roll << std::endl;

	yaw += degree;

// 	m_boat->setQuaternian(m_boat->getQuaternian(yaw, pitch, roll));
// 	m_boat->getEulerAngle(yaw, pitch, roll);

// 	yaw = 0.1f;
// 	pitch = 0.2f;
// 	roll = 0.1f;
	m_boat->setQuaternian(m_boat->getQuaternian(yaw, pitch, roll));

 	float newY, newP, newR;
 	m_boat->getEulerAngle(newY, newP, newR);
	std::cout << "After: yaw: " << newY << " Pitch: " << newP << " roll: " << newR << std::endl;
// 	glm::vec3 dif = glm::vec3(newY - yaw, newP - pitch, newR - roll);
// 	if (glm::length(dif) > EPSILON)
// 	{
// 		printf("*************Rotation Error: %f \n" , glm::length(dif));
// 	}

}

void Coupling::propel(float acceleration)
{
	m_boat->setImpetus(acceleration);
}

float2 Coupling::getLocalBoatCenter()
{
	glm::vec3 center = m_boat->getCenter();
	float2 waveOrigin = m_trail->getOrigin();
	float dg = m_trail->getRealGridSize();
	int resolution = m_trail->getGridSize();
	return make_float2((center.x - waveOrigin.x) / dg, (center.z - waveOrigin.y) / dg)/resolution;
}

RigidBody* Coupling::getBoat()
{
	return m_boat;
}

CapillaryWave* Coupling::getTrail()
{
	return m_trail;
}

void Coupling::initialize(RigidBody * boat, CapillaryWave* wave)
{
	m_trail = wave;
	m_boat = boat;

	int sizeInBytes = boat->getSamplingPointSize() * sizeof(float3);
	int sizeInBytesF = boat->getSamplingPointSize() * sizeof(float);

	m_reduce = Physika::Reduction<float>::Create(boat->getSamplingPointSize());

	cudaMalloc(&m_forceX, sizeInBytesF);
	cudaMalloc(&m_forceY, sizeInBytesF);
	cudaMalloc(&m_forceZ, sizeInBytesF);
	cudaMalloc(&m_torqueX, sizeInBytesF);
	cudaMalloc(&m_torqueY, sizeInBytesF);
	cudaMalloc(&m_torqueZ, sizeInBytesF);

	cudaMalloc(&m_sample_heights, sizeInBytesF);

	glm::vec3 center = boat->getCenter();

	float dg = m_trail->getRealGridSize();

	int nx = center.x / dg - m_trail->getGridSize() / 2;
	int ny = center.z / dg - m_trail->getGridSize() / 2;

	m_trail->setOriginX(nx);
	m_trail->setOriginY(ny);
}

void Coupling::initialize(RigidBody * boat, PhaseField* wave)
{
	m_phasefield = wave;
	m_boat = boat;

	int sizeInBytes = boat->getSamplingPointSize() * sizeof(float3);
	int sizeInBytesF = boat->getSamplingPointSize() * sizeof(float);

	m_reduce = Physika::Reduction<float>::Create(boat->getSamplingPointSize());

	cudaMalloc(&m_forceX, sizeInBytesF);
	cudaMalloc(&m_forceY, sizeInBytesF);
	cudaMalloc(&m_forceZ, sizeInBytesF);
	cudaMalloc(&m_torqueX, sizeInBytesF);
	cudaMalloc(&m_torqueY, sizeInBytesF);
	cudaMalloc(&m_torqueZ, sizeInBytesF);

	cudaMalloc(&m_sample_heights, sizeInBytesF);

	glm::vec3 center = boat->getCenter();//0,0,0

	float dg = m_phasefield->getRealGridSize();//512

	int nx = center.x / dg - m_phasefield->getGridSize() / 2;//getGridSize()=512
	int ny = center.z / dg - m_phasefield->getGridSize() / 2;

	m_phasefield->setOriginX(nx);
	m_phasefield->setOriginY(ny);
}