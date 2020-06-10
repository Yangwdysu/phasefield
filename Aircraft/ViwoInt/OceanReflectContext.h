#pragma once

#include "GLObjects.h"

#include "OceanVec.h"


#include <Camera.h>

class OceanReflectContext
{
public:
	void allocBuffers(int width, int height);

	/// \brief ���ɻ��Ʒ����������õľ���
	/// \note ����������޸� OpenGL ������״̬��ʹ�� get*Matrix() ��ȡ��������
	void makeMatrices(OceanVec::dvec3 const& eye, OceanVec::dvec3 const& direction, OceanVec::dvec3 const& up, OceanVec::dvec3 const& surface_base, OceanVec::dvec3 const& surface_normal, OceanVec::dmat4 const& proj_matrix);
	void makeReflectionCamera( const CCamera &origin_camera, OceanVec::dvec3 const& surface_base, OceanVec::dvec3 const& surface_normal, CCamera &reflection_camera ) const;
	OceanVec::dmat4 const& getModelviewMatrix() const;
	OceanVec::dmat4 const& getProjectionMatrix() const;
	GLuint getFBO() const;
	GLuint getColorTexture() const;
	GLuint getDepthTexture() const;

	void bindColorTexture(GLuint unit) const;
	void bindRemoteDepthTexture(GLuint unit) const;
	void bindNearbyDepthTexture(GLuint unit) const;
	void bindRawDepthTexture(GLuint unit) const;

private:
	OceanVec::dmat4 m_Modelview, m_Projection;

	GLFramebufferObject m_FBO;
	GLTextureObject m_ReflectColorTex, m_ReflectDepthTex;
};
