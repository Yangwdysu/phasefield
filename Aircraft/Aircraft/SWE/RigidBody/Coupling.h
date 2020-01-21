#pragma once
#include "quaternion.h"
#include <vector>
#include <iostream>
#include <string.h>

#include "../CapillaryWave.h"
#include "../OceanPatch.h"
#include "Reduction.h"
#include"PhaseField.h"
namespace WetBrush {
	class RigidBody;

	class Coupling
	{
	public:
		Coupling(OceanPatch* ocean_patch);
		~Coupling();

		void initialize(RigidBody *boat, CapillaryWave* wave);
		void initialize(RigidBody * boat, PhaseField* wave);
		void animate(float dt);

		void setHeightShift(float shift);

		void setBoatMatrix(glm::dmat4 mat, float dt);
		glm::dmat4 getBoatMatrix();

		void steer(float degree);
		void propel(float acceleration);

		float2 getLocalBoatCenter();

		RigidBody* getBoat();
		CapillaryWave* getTrail();

		void setName(std::string name) { m_name = name; }
	private:
		glm::vec3 m_prePos;

		std::string m_name;
		float m_heightShift;
		OceanPatch* m_ocean_patch;					//fft patch

		float m_eclipsedTime;

		//	float3* m_oceanCentroid;			//fft displacement at the center of boat

		float* m_forceX;					//forces at sample points
		float* m_forceY;
		float* m_forceZ;

		float* m_torqueX;					//torques at sample points
		float* m_torqueY;
		float* m_torqueZ;

		//采样点对应的海面高度
		float* m_sample_heights;

		bool m_force_corrected;

		glm::vec3 m_force_corrector;
		glm::vec3 m_torque_corrector;

		float2 m_origin;

		RigidBody* m_boat;
		CapillaryWave* m_trail;
		PhaseField* m_phasefield;
		float m_heightScale = 0.2f;

		Physika::Reduction<float>* m_reduce;
	};
}