#include "RigidBody.h"
#include <cuda_runtime.h>
#include "../cuda_helper_math.h"
#include<iostream>
#include <fstream>
#include <math.h>
#include "../types.h"
namespace WetBrush {
	RigidBody::RigidBody()
	{
		m_mass = 1.0f;
		scale.x = 0.1;
		scale.y = 0.1;
		scale.z = 0.1;
		m_yaw = 0.0f;
		m_pitch = 0.0f;
		m_roll = 0.0f;
		m_acceleration = 0.0f;
		m_damping = 0.9f;
		m_recoverSpeed = 0.3f;
		m_maxTransVel = 10.0f;
		m_maxAngularVel = 3.0f;
		m_renderReady = false;
		m_inverseInertia = glm::mat3();
		m_center = glm::vec3(0);
		m_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
		m_angularvelocity = glm::vec3(0.0f, 0.0f, 0.0f);
		m_quaternion = Quaternion<float>(1.0f, 0.0f, 0.0f, 0.0f);
	}

	void RigidBody::loadForcePoints(const char* path)
	{
		std::ifstream points_stream(path);
		if (!points_stream.is_open())
		{
			std::cout << "ERROR::IFSTREAM:: Can not open file: " << path << std::endl;
		}

		float tmpxMin = 999999.0, tmpyMin = 999999.0, tmpzMin = 999999.0;
		float tmpxMax = -999999.0, tmpyMax = -999999.0, tmpzMax = -999999.0;

		int bj1 = 0;
		int bj2 = 0;

		std::string str;
		while (points_stream >> str)
		{
			if (std::string("v") == str)
			{
				float value1, value2, value3;

				points_stream >> value1;
				points_stream >> value2;
				points_stream >> value3;

				float3 in;
				in.x = value1;
				in.y = value2;
				in.z = value3;

				samples.push_back(glm::vec3(value1, value2, value3));
			}
			else if (std::string("vn") == str)
			{
				float value1, value2, value3;

				points_stream >> value1;
				points_stream >> value2;
				points_stream >> value3;

				float square = value1*value1 + value2*value2 + value3*value3;
				float len = sqrtf(square);
				value1 /= len;
				value2 /= len;
				value3 /= len;

				normals.push_back(glm::vec3(value1, value2, value3));
			}
		}

		m_numOfSamples = samples.size();
		//cout << m_numOfSamples << endl;

		int sizeInBytes = m_numOfSamples * sizeof(float3);
		cudaMalloc(&m_deviceSamples, sizeInBytes);
		cudaMemcpy(m_deviceSamples, &samples[0], sizeInBytes, cudaMemcpyHostToDevice);

		cudaMalloc(&m_deviceNormals, sizeInBytes);
		cudaMemcpy(m_deviceNormals, &normals[0], sizeInBytes, cudaMemcpyHostToDevice);
	}

	void RigidBody::update(glm::vec3 force, glm::vec3 torque, float dt)
	{
		m_velocity.x = m_velocity.x < m_maxTransVel ? m_velocity.x : m_maxTransVel;
		m_velocity.x = m_velocity.x > -m_maxTransVel ? m_velocity.x : -m_maxTransVel;

		m_velocity.z = m_velocity.z < m_maxTransVel ? m_velocity.z : m_maxTransVel;
		m_velocity.z = m_velocity.z > -m_maxTransVel ? m_velocity.z : -m_maxTransVel;

		m_angularvelocity.x = m_angularvelocity.x < m_maxAngularVel ? m_angularvelocity.x : m_maxAngularVel;
		m_angularvelocity.x = m_angularvelocity.x > -m_maxAngularVel ? m_angularvelocity.x : -m_maxAngularVel;
		m_angularvelocity.y = m_angularvelocity.y < m_maxAngularVel ? m_angularvelocity.y : m_maxAngularVel;
		m_angularvelocity.y = m_angularvelocity.y > -m_maxAngularVel ? m_angularvelocity.y : -m_maxAngularVel;
		m_angularvelocity.z = m_angularvelocity.z < m_maxAngularVel ? m_angularvelocity.z : m_maxAngularVel;
		m_angularvelocity.z = m_angularvelocity.z > -m_maxAngularVel ? m_angularvelocity.z : -m_maxAngularVel;


		glm::mat3 rot = getOrientation();
		m_velocity += dt*force / m_mass + dt * m_acceleration * rot * glm::vec3(0.0f, 0.0f, -1.0f);
		m_angularvelocity += dt*m_inverseInertia*glm::transpose(rot)*torque;

		glm::vec3 local_v = glm::transpose(rot)*m_velocity;
		local_v.x *= 0.5f;
		local_v.z *= m_damping;

		m_velocity = rot*local_v;
		m_angularvelocity *= m_damping;
	}

	void RigidBody::advect(float dt)
	{
		m_center += dt*(m_velocity);

		// update rotation
		glm::vec3 angularVel = m_angularvelocity;
		m_quaternion = m_quaternion + (0.5f * dt) * Quaternion<float>(0, angularVel.x, angularVel.y, angularVel.z) * m_quaternion;

		m_quaternion.Normalize();
		recoverPose();
	}

	void RigidBody::recoverPose()
	{
		getEulerAngle(m_yaw, m_pitch, m_roll);
		m_roll *= (1.0f - m_recoverSpeed);
		m_pitch *= (1.0f - m_recoverSpeed);
		m_quaternion = getQuaternian(m_yaw, m_pitch, m_roll);
	}

	void RigidBody::setCenter(glm::vec3 translation)
	{
		m_center = translation;
	}

	void RigidBody::setMass(float mass)
	{
		m_mass = mass;
	}

	void RigidBody::setInertiaTensor(glm::mat3 inertia)
	{
		m_inverseInertia = glm::inverse(inertia);
	}

	void RigidBody::setInverseInertiaTensor(glm::mat3 invInertia)
	{
		m_inverseInertia = invInertia;
	}

	void RigidBody::initialize(std::string obj_name, std::string sample_file)
	{
		//sdf.ReadSDF(sdf_name);
		m_mesh.loadObj(obj_name);
		loadForcePoints(sample_file.c_str());
	}

	void RigidBody::initRenderingContext()
	{
		std::vector<float3>& vert = m_mesh.getVertices();
		std::vector<int3>& faces = m_mesh.getFaces();

		glGenBuffers(1, &m_vertexBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
		glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof(float3), &vert[0], GL_STATIC_DRAW);

		glGenBuffers(1, &m_indexBuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * faces.size() * sizeof(int), &faces[0], GL_STATIC_DRAW);

		m_renderReady = true;
	}

	void RigidBody::display()
	{
		if (!m_renderReady)
		{
			initRenderingContext();
		}

		float angle;
		float axis[3];

		glm::mat3 glmMat = getOrientation();
		GLfloat rot[16] = { glmMat[0][0], glmMat[0][1], glmMat[0][2], 0,
							glmMat[1][0], glmMat[1][1], glmMat[1][2], 0,
							glmMat[2][0], glmMat[2][1], glmMat[2][2], 0,
							0,0,0,1 };

		glPushMatrix();

		glEnableClientState(GL_INDEX_ARRAY);
		glEnableClientState(GL_VERTEX_ARRAY);

		glScalef(scale.x, scale.y, scale.z);
		glTranslatef(m_center.x, m_center.y, m_center.z);
		glMultMatrixf(rot);

		glBegin(GL_LINES);
		for (int i = 0; i < samples.size(); i++)
		{
			glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
			glVertex3f(samples[i].x, samples[i].y, samples[i].z);
			glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
			glVertex3f(samples[i].x + normals[i].x, samples[i].y + normals[i].y, samples[i].z + normals[i].z);
		}
		glEnd();

		glColor4f(1.0f, 1.0f, 1.0f, 0.5f);
		glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
		glVertexPointer(3, GL_FLOAT, sizeof(float3), 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
		glIndexPointer(GL_INT, 0, 0);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDrawElements(GL_TRIANGLES, 3 * m_mesh.getFaces().size(), GL_UNSIGNED_INT, 0);
		glDisable(GL_BLEND);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_INDEX_ARRAY);
		glPopMatrix();
	}

	glm::mat3 RigidBody::getOrientation()
	{
		glm::mat3 rot;
		Quaternion<float> quat = Quaternion<float>(m_quaternion.Gets(), m_quaternion.Getx(), m_quaternion.Getz(), m_quaternion.Gety());
		quat.Quaternion2Matrix(&rot[0][0]);//四元数转旋转矩阵
		return rot;//返回一个旋转矩阵
	}


	//四元数转欧拉角
	void RigidBody::getEulerAngle(float& yaw, float& pitch, float& roll)
	{
		//该系统实现y轴朝上，标准yaw, pitch, roll则是z轴朝下，因为计算之前需要先绕着x轴旋转90度。
		Quaternion<float> quat = m_quaternion*Quaternion<float>(cos(M_PI / 4.0f), sin(M_PI / 4.0f), 0.0f, 0.0f);
		// roll (x-axis rotation)
		double sinr_cosp = +2.0 * (quat.Gets() * quat.Getx() + quat.Gety() * quat.Getz());
		double cosr_cosp = +1.0 - 2.0 * (quat.Getx() * quat.Getx() + quat.Gety() * quat.Gety());
		//减去90度
		roll = atan2(sinr_cosp, cosr_cosp) - M_PI / 2.0f;

		// pitch (y-axis rotation)
		double sinp = +2.0 * (quat.Gets() * quat.Gety() - quat.Getz() * quat.Getx());
		if (fabs(sinp) >= 1)
			pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
		else
			pitch = asin(sinp);

		// yaw (z-axis rotation)
		double siny_cosp = +2.0 * (quat.Gets() * quat.Getz() + quat.Getx() * quat.Gety());
		double cosy_cosp = +1.0 - 2.0 * (quat.Gety() * quat.Gety() + quat.Getz() * quat.Getz());
		yaw = atan2(siny_cosp, cosy_cosp);
	}

	//欧拉角转四元数
	Quaternion<float> RigidBody::getQuaternian(float yaw, float pitch, float roll)
	{
		double cy = cos(yaw * 0.5);
		double sy = sin(yaw * 0.5);
		double cp = cos(pitch * 0.5);
		double sp = sin(pitch * 0.5);
		double cr = cos(roll * 0.5);
		double sr = sin(roll * 0.5);

		float w = cy * cp * cr + sy * sp * sr;
		float x = cy * cp * sr - sy * sp * cr;
		float y = sy * cp * sr + cy * sp * cr;
		float z = sy * cp * cr - cy * sp * sr;

		return Quaternion<float>(w, x, y, z);
	}

	void RigidBody::setQuaternian(Quaternion<float> quat)
	{
		m_quaternion = quat;
	}

	void RigidBody::setImpetus(float acc)
	{
		m_acceleration = acc;
	}

	void RigidBody::setVelocityDamping(float d)
	{
		m_damping = d;
	}
}
