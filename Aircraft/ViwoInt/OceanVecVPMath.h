#pragma once

// #include <glm/glm.hpp>
// #include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtc/type_ptr.hpp>
#include "VPMath.h"
#define M_PI       3.14159265358979323846
namespace OceanVec
{

	//using namespace glm;
	typedef VPE::Basic::CVector2 vec2;
	typedef VPE::Basic::CDVector2 dvec2;
	typedef VPE::Basic::CVector3 vec3;
	typedef VPE::Basic::CQuaternion vec4;
	typedef VPE::Basic::CDVector3 dvec3;
	typedef VPE::Basic::CDQuaternion dvec4;

	typedef VPE::Basic::CFMatrix4X4 mat4;

	typedef VPE::Basic::CDMatrix3X3 dmat3;
	typedef VPE::Basic::CDMatrix4X4 dmat4;

	struct mat3x4
	{
	public:
		mat3x4(){}
		mat3x4(dmat3 const& m)
		{
			col[0] = vec4(m.m_fMat11, m.m_fMat12, m.m_fMat13, 0.0);
			col[1] = vec4(m.m_fMat21, m.m_fMat22, m.m_fMat23, 0.0);
			col[2] = vec4(m.m_fMat31, m.m_fMat32, m.m_fMat33, 0.0);
		}

		vec4& operator [] (std::ptrdiff_t i)
		{
			return col[i];
		}

		vec4 const& operator [] (std::ptrdiff_t i) const
		{
			return col[i];
		}

	private:
		vec4 col[3];
	};

	inline float radians(float deg)
	{
		return Deg2Rad(deg);
	}

	inline double radians(double deg)
	{
		return Deg2Rad(deg);
	}

	inline float dot(vec2 const& a, vec2 const& b){
		return a.Dot(b);
	}
	inline double dot(dvec2 const& a, dvec2 const& b)
	{
		return a.Dot(b);
	}
	inline float dot(vec3 const& a, vec3 const& b){
		return a.Dot(b);
	}
	inline double dot(dvec3 const& a, dvec3 const& b){
		return a.Dot(b);
	}

	inline double length(dvec3 const& v)
	{
		return v.GetLength();
	}

	inline double distance(dvec3 const& a, dvec3 const& b)
	{
		return (a - b).GetLength();
	}

	inline vec3 swizzle_xyz(vec4 const& v)
	{
		return vec3(v.x, v.y, v.z);
	}

	inline dvec3 swizzle_xyz(dvec4 const& v)
	{
		return dvec3(v.x, v.y, v.z);
	}

	inline vec3 div_w(vec4 const& v)
	{
		return swizzle_xyz(v) * (1.0f / v.w);
	}

	inline dvec3 div_w(dvec4 const& v)
	{
		return swizzle_xyz(v) * (1.0 / v.w);
	}

	inline vec3 mulv3(mat4 const& m, vec3 const& v)
	{
		return div_w(m.Multiply(OceanVec::vec4(v.x,v.y,v.z, 1.0f)));
	}

	inline dvec3 mulv3(dmat4 const& m, dvec3 const& v)
	{
		return div_w(m.Multiply(OceanVec::dvec4(v.x,v.y,v.z, 1.0)));
	}

	inline dvec4 mul_dmat3x4_dvec3(dmat4 const& m, dvec3 const& v)
	{
		return dvec4(m.m_fMatrix[0]) * v.x + dvec4(m.m_fMatrix[1]) * v.y + dvec4(m.m_fMatrix[2]) * v.z;
	}

	template <typename Value>
	inline ::VPE::Basic::_TMatrix4X4<Value> lookAtDir(
		::VPE::Basic::_TVector3<Value> const& eye,
		::VPE::Basic::_TVector3<Value> const& dir,
		::VPE::Basic::_TVector3<Value> const& up)
	{
		::VPE::Basic::_TMatrix4X4<Value> temp;
		temp.Identity();
		temp.look(eye,dir,up);
		return temp;
	}

	template <typename Value>
	inline ::VPE::Basic::_TMatrix4X4<Value> perspectiveFov(
		const Value fovy,
		const Value width,
		const Value height,
		const Value zNear,
		const Value zFar)
	{
		::VPE::Basic::_TMatrix4X4<Value> result;
		result.buildProjectionMatrixPerspectiveRH2(fovy, width / height, zNear, zFar);
		return result;
	}

	/// m 无效， 用在左乘
	inline mat4 scale(mat4 const& m, vec3 const& s)
	{
		mat4 temp;
		temp.Identity();
		temp.SetScale(s);
		return temp;
	}

	inline vec2 normalize(vec2 a){
		a.Normalize();
		return a;
	}
	inline dvec2 normalize(dvec2 a) {
		a.Normalize();
		return a;
	}

	inline dvec3 normalize(dvec3 a){
		a.Normalize();
		return a;
	}

	inline dmat4 inverse(dmat4 const& m)
	{
		return m.inverse();
	}

	inline double const* value_ptr(dmat4 const& m)
	{
		return m.Get();
	}

	inline dvec3 cross(dvec3 const& a, dvec3 const& b)
	{
		return a.crossProduct(b);
	}

	template <typename T, typename U>
	inline T mix(T a, T b, U s)
	{
		return s*a + (U(1)-s) * b;
	}
}
