#include "SurfaceMesh.h"
#include <fstream>
#include <iostream>
#include <assert.h>
#include "../cuda_helper_math.h"

SurfaceMesh::SurfaceMesh()
{
	_m = 0;
	_Cx = 0;
	_Cy = 0;
	_Cz = 0;
}

SurfaceMesh::~SurfaceMesh()
{
}

void SurfaceMesh::loadObj(std::string filename)
{
	std::ifstream input(filename, std::ios::in);
	if (!input.is_open()) {
		std::cout << "load obj error:" << filename << std::endl;
		assert(0);
	}
	float x, y, z;
	int v1, v2, v3;
	std::string keyword;
	std::string str_dum;
	int v_num = 0;
	int f_num = 0;

	float maxx = -10000;
	float maxy = -10000;
	float maxz = -10000;
	float minx = 10000;
	float miny = 10000;
	float minz = 10000;

	while (!input.eof())
	{
		input >> keyword;
		if (keyword == std::string("v"))
		{
			v_num++;
			input >> x;
			input >> y;
			input >> z;
			m_vertices.push_back(make_float3(x, y, z));
			maxx = max(x, maxx);
			maxy = max(y, maxy);
			maxz = max(z, maxz);
			minx = min(x, minx);
			miny = min(y, miny);
			minz = min(z, minz);
		}
		else if (keyword == std::string("f"))
		{
			f_num++;
			input >> v1;	//input >> str_dum;
			input >> v2;	//input >> str_dum;
			input >> v3;
			m_faces.push_back({ v1 - 1, v2 - 1, v3 - 1 });
		}
		else
			continue;
	}
}
void SurfaceMesh::AddTriangleContribution(
	double x1, double y1, double z1,    // Triangle's vertex 1
	double x2, double y2, double z2,    // Triangle's vertex 2
	double x3, double y3, double z3)    // Triangle's vertex 3
{
	// Signed volume of this tetrahedron.
	double v = x1*y2*z3 + y1*z2*x3 + x2*y3*z1 -
		(x3*y2*z1 + x2*y1*z3 + y3*z2*x1);
	// Contribution to the mass
	_m += v;

	// Contribution to the centroid
	double x4 = x1 + x2 + x3;           _Cx += (v * x4);
	double y4 = y1 + y2 + y3;           _Cy += (v * y4);
	double z4 = z1 + z2 + z3;           _Cz += (v * z4);

	// Contribution to moment of inertia monomials
	_xx += v * (x1*x1 + x2*x2 + x3*x3 + x4*x4);
	_yy += v * (y1*y1 + y2*y2 + y3*y3 + y4*y4);
	_zz += v * (z1*z1 + z2*z2 + z3*z3 + z4*z4);
	_yx += v * (y1*x1 + y2*x2 + y3*x3 + y4*x4);
	_zx += v * (z1*x1 + z2*x2 + z3*x3 + z4*x4);
	_zy += v * (z1*y1 + z2*y2 + z3*y3 + z4*y4);
}
void SurfaceMesh::computeInertiaTensor(float3 scale, float* re)
{
	for (auto &i : m_faces)
	{
		AddTriangleContribution(scale.x*m_vertices[i.x].x, scale.y*m_vertices[i.x].y, scale.z*m_vertices[i.x].z, scale.x*m_vertices[i.y].x, scale.y*m_vertices[i.y].y, scale.z*m_vertices[i.y].z, scale.x*m_vertices[i.z].x, scale.y*m_vertices[i.z].y, scale.z*m_vertices[i.z].z);
	}
	double r = 1.0 / (4 * _m);

	double Ixx, Iyy, Izz, Iyx, Izx, Izy;

	Cx = _Cx * r;
	Cy = _Cy * r;
	Cz = _Cz * r;

	// Mass
	m = _m / 6;

	// Moment of inertia about the centroid.
	r = 1.0 / 120;
	Iyx = _yx * r - m * Cy*Cx;
	Izx = _zx * r - m * Cz*Cx;
	Izy = _zy * r - m * Cz*Cy;

	_xx = _xx * r - m * Cx*Cx;
	_yy = _yy * r - m * Cy*Cy;
	_zz = _zz * r - m * Cz*Cz;

	Ixx = _yy + _zz;
	Iyy = _zz + _xx;
	Izz = _xx + _yy;

	re[0] = Ixx;
	re[1] = -Iyx;
	re[2] = -Izx;
	re[3] = -Iyx;
	re[4] = Iyy;
	re[5] = -Izy;
	re[6] = -Izx;
	re[7] = -Izy;
	re[8] = Izz;

	//printf("%f %f %f\n", Cx, Cy, Cz);
}

