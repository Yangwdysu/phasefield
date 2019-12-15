#pragma once
//#include "DistanceField3D.h"
#define GLM_FORCE_PURE
#include "quaternion.h"
#include <GL/glew.h>
#include <vector>
#include <iostream>
#include "SurfaceMesh.h"
#include <glm/glm.hpp>

#define jia 1.57


class RigidBody
{
public:
	RigidBody();
	~RigidBody() {};

	void update(glm::vec3 force, glm::vec3 torque, float dt);

	void advect(float dt);

	void recoverPose();

	void setCenter(glm::vec3 translation);
	void setVelocity(glm::vec3 vel) { m_velocity = vel; }
	void setMass(float mass);
	void setInertiaTensor(glm::mat3 inertia);
	void setInverseInertiaTensor(glm::mat3 invInertia);

	void initialize(std::string obj_name, std::string sample_file);
	void initRenderingContext();

	void loadForcePoints(const char* path);

	void display();

	int getSamplingPointSize() { return m_numOfSamples; }

	float3* getSamples() { return m_deviceSamples; }
	float3* getNormals() { return m_deviceNormals; }
	glm::vec3 getVelocity() { return m_velocity; }
	glm::vec3 getAngularVelocity() { return m_angularvelocity; }
	glm::vec3 getCenter() { return m_center; }
	glm::mat3 getOrientation();
	
	float getYaw() { return m_yaw; }
	float getPitch() { return m_pitch; }
	float getRoll() { return m_roll; }

	//yaw: zÖáÐý×ª£»pitch£ºyÖáÐý×ª£»roll£ºxÖáÐý×ª
	void getEulerAngle(float& yaw, float& pitch, float& roll);
	Quaternion<float> getQuaternian(float yaw, float pitch, float roll);
	void setQuaternian(Quaternion<float> quat);
	Quaternion<float> getQuaternian() { return m_quaternion; }

	void setImpetus(float acc);
	void setVelocityDamping(float d);

public:
	float3 scale;

	bool m_renderReady;
	int m_numOfSamples;

	float m_mass;
	float3* m_deviceSamples;
	float3* m_deviceNormals;

	std::vector<glm::vec3> samples;
	std::vector<glm::vec3> normals;

	float m_recoverSpeed;

	glm::vec3 m_center;
	glm::mat3 m_inverseInertia;

	glm::vec3 m_velocity;

	glm::vec3 m_angularvelocity;

	float m_acceleration;
	float m_damping;

	float m_maxTransVel;
	float m_maxAngularVel;

	float m_yaw;
	float m_pitch;
	float m_roll;

	Quaternion<float> m_quaternion;

	GLuint m_vertexBuffer;
	GLuint m_indexBuffer;

	SurfaceMesh m_mesh;
};