#pragma once

#include "OceanVec.h"

class OceanRefCoord
{
public:
	OceanRefCoord(OceanVec::dvec3 const& init_center = OceanVec::dvec3(1., 0., 0.), double grid_size = 2000.0, double radius = 6378137.);

	void MoveFocus(OceanVec::dvec3 const& new_focus_);

	OceanVec::dvec3 const& GetCenter() const;
	OceanVec::dvec3 const& GetEast() const;
	OceanVec::dvec3 const& GetNorth() const;
	OceanVec::dvec3 const& GetUp() const;

	/// 仅用于调试
	void SetDebugCenter(OceanVec::dvec3 const& center);

private:
	OceanRefCoord(OceanRefCoord const&) = delete;
	OceanRefCoord& operator = (OceanRefCoord const&) = delete;

	void SetCenter(OceanVec::dvec3 const& center);

	OceanVec::dvec3 SurfacePoint(OceanVec::dvec3 const& dir) const;

	double const m_grid_size, m_radius;
	OceanVec::dvec3 m_center, m_east, m_north, m_up;
	OceanVec::dvec3 m_neighbour_nodes[8];
};
