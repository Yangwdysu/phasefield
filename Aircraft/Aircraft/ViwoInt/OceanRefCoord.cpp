#include "OceanRefCoord.h"

#include <cstdio>

#include <limits>

#include "CoordUtils/LocalCoord.h"

OceanRefCoord::OceanRefCoord(OceanVec::dvec3 const& init_center, double grid_size, double radius): m_grid_size(grid_size), m_radius(radius)
{
	SetCenter(init_center);
}

void OceanRefCoord::MoveFocus(OceanVec::dvec3 const& new_focus_)
{
	OceanVec::dvec3 const new_focus = SurfacePoint(new_focus_);
	double const new_focus_distance = OceanVec::distance(new_focus, m_center);

	if(new_focus_distance < m_grid_size * 0.75)
		return;

	if(new_focus_distance >= m_grid_size * 1.5)
	{
		SetCenter(new_focus);
	}
	else
	{
		int nearest_neighbour = 0;
		double nearest_distance = std::numeric_limits<double>::infinity();
		for(int i = 0; i < 7; ++i)
		{
			double const dist = OceanVec::distance(new_focus, m_neighbour_nodes[i]);
			if(dist < nearest_distance)
			{
				nearest_neighbour = i;
				nearest_distance = dist;
			}
		}
		SetCenter(m_neighbour_nodes[nearest_neighbour]);
	}
}

OceanVec::dvec3 const& OceanRefCoord::GetCenter() const
{
	return m_center;
}

OceanVec::dvec3 const& OceanRefCoord::GetEast() const
{
	return m_east;
}

OceanVec::dvec3 const& OceanRefCoord::GetNorth() const
{
	return m_north;
}

OceanVec::dvec3 const& OceanRefCoord::GetUp() const
{
	return m_up;
}

void OceanRefCoord::SetDebugCenter(OceanVec::dvec3 const& center)
{
	SetCenter(center);
}

void OceanRefCoord::SetCenter(OceanVec::dvec3 const& center)
{
	using namespace OceanVec;
	//std::fprintf(stderr, "Recentering: (%f, %f, %f)\n", center.x, center.y, center.z);
	GenerateLocalDirections(center, m_east, m_north, m_up);
	m_center = m_up * m_radius;
	dvec3 const scaled_east = m_east * m_grid_size;
	dvec3 const scaled_north = m_north * m_grid_size;
	m_neighbour_nodes[0] = SurfacePoint(m_center - scaled_east);
	m_neighbour_nodes[1] = SurfacePoint(m_center - scaled_east - scaled_north);
	m_neighbour_nodes[2] = SurfacePoint(m_center - scaled_north);
	m_neighbour_nodes[3] = SurfacePoint(m_center + scaled_east - scaled_north);
	m_neighbour_nodes[4] = SurfacePoint(m_center + scaled_east);
	m_neighbour_nodes[5] = SurfacePoint(m_center + scaled_east + scaled_north);
	m_neighbour_nodes[6] = SurfacePoint(m_center + scaled_north);
	m_neighbour_nodes[7] = SurfacePoint(m_center - scaled_east + scaled_north);
}

OceanVec::dvec3 OceanRefCoord::SurfacePoint(OceanVec::dvec3 const& dir) const
{
	return OceanVec::normalize(dir) * m_radius;
}
