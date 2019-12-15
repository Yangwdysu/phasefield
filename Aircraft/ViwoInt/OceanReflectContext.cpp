#include "OceanReflectContext.h"

#include "GLSamplerManager.h"
#include "GLTextureUtils.h"

#include "glenv.h"

#include "ViwoUtils.h"

using namespace OceanVec;

static dvec3 plane_reflect_point(dvec3 const& P, dvec3 const& plane_base, dvec3 const& normal)
{
	double const k = dot(P - plane_base, normal);
	return P - normal * (2 * k);
}

static dvec3 plane_reflect_dir(dvec3 const& D, dvec3 const& normal)
{
	double const k = dot(D, normal);
	return D - normal * (2 * k);
}

void OceanReflectContext::allocBuffers( int width, int height )
{
	m_FBO.allocIfNull();
	m_ReflectColorTex.allocIfNull();
	m_ReflectDepthTex.allocIfNull();

	gl_Call(glBindTexture(GL_TEXTURE_2D, m_ReflectColorTex.get()));
	gl_Call(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));

	gl_Call(glBindTexture(GL_TEXTURE_2D, m_ReflectDepthTex.get()));
	gl_Call(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL));

	ViwoUtils::PushDrawFrambuffer( m_FBO.get() );
	gl_Call(glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_ReflectColorTex.get(), 0));
	gl_Call(glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_ReflectDepthTex.get(), 0));
	ViwoUtils::PopDrawFramebuffer();
}

void OceanReflectContext::makeMatrices( OceanVec::dvec3 const& eye, OceanVec::dvec3 const& direction, OceanVec::dvec3 const& up, OceanVec::dvec3 const& surface_base, OceanVec::dvec3 const& surface_normal, OceanVec::dmat4 const& proj_matrix )
{
	dvec3 const r_eye = plane_reflect_point(eye, surface_base, surface_normal);
	dvec3 const r_direction = plane_reflect_dir(direction, surface_normal);
	dvec3 const r_up = plane_reflect_dir(up, surface_normal);

	m_Modelview = lookAtDir(r_eye, r_direction, r_up);
	m_Projection = scale(dmat4(), dvec3(-1., 1., 1.)) * proj_matrix;
}

dmat4 const& OceanReflectContext::getModelviewMatrix() const
{
	return m_Modelview;
}

dmat4 const& OceanReflectContext::getProjectionMatrix() const
{
	return m_Projection;
}

GLuint OceanReflectContext::getFBO() const
{
	return m_FBO.get();
}

GLuint OceanReflectContext::getColorTexture() const
{
	return m_ReflectColorTex.get();
}

GLuint OceanReflectContext::getDepthTexture() const
{
	return m_ReflectDepthTex.get();
}

void OceanReflectContext::bindColorTexture( GLuint unit ) const
{
	bindTextureUnit(unit, GL_TEXTURE_2D, m_ReflectColorTex, GLSamplerManager::GetLinearSampler());
}

void OceanReflectContext::bindRemoteDepthTexture( GLuint unit ) const
{
	bindTextureUnit(unit, GL_TEXTURE_2D, m_ReflectDepthTex, GLSamplerManager::GetLinearEqualSampler());
}

void OceanReflectContext::bindNearbyDepthTexture( GLuint unit ) const
{
	bindTextureUnit(unit, GL_TEXTURE_2D, m_ReflectDepthTex, GLSamplerManager::GetLinearGreaterSampler());
}

void OceanReflectContext::bindRawDepthTexture( GLuint unit ) const
{
	bindTextureUnit(unit, GL_TEXTURE_2D, m_ReflectDepthTex, GLSamplerManager::GetNearestSampler());
}


void OceanReflectContext::makeReflectionCamera( const CCamera &origin_camera, OceanVec::dvec3 const& surface_base, OceanVec::dvec3 const& surface_normal, CCamera &reflection_camera ) const
{
	dvec3 const eye = origin_camera.getPosition();
	dvec3 const direction = origin_camera.getDirection();
	dvec3 const up = origin_camera.getUp();

	dvec3 const r_eye = plane_reflect_point(eye, surface_base, surface_normal);
	dvec3 const r_direction = plane_reflect_dir(direction, surface_normal);
	dvec3 const r_up = plane_reflect_dir(up, surface_normal);

	reflection_camera.LookAt( r_eye, r_eye + r_direction * 1000.0 , r_up );
}
