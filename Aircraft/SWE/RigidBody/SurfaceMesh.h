#pragma once
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <vector>

class SurfaceMesh
{
public:
	SurfaceMesh();
	~SurfaceMesh();

	void loadObj(std::string filename);

	std::vector<float3>& getVertices() { return m_vertices; }
	std::vector<int3>& getFaces() { return m_faces; }
	void computeInertiaTensor(float3 scale, float* re);
	double Cx, Cy, Cz;                   // Centroid
	double m;
private:
	std::vector<float3> m_vertices;
	std::vector<int3> m_faces;

	void AddTriangleContribution(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3);
	double _m;                              // Mass
	double _Cx, _Cy, _Cz;
	double _xx, _yy, _zz, _yx, _zx, _zy;    // Moment of inertia tensor
};
