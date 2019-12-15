#pragma once
#include<glm/glm.hpp>

class World2Local {
public:
	
	static glm::vec2 World2Local::world2LocalSurface(glm::dvec3 p);

	static double s_earth_radius;
	static glm::dvec3 s_center;
	static glm::dvec3 s_east;
	static glm::dvec3 s_north;
	static glm::dvec3 s_surface_up;

	static glm::dmat4 s_world2local;
	static glm::dmat4 s_local2world;
	static glm::dmat4 s_local2simulation;
	static glm::dmat4 s_simulation2local;
	static glm::dmat4 s_simulation2world;
	static glm::dmat4 s_world2simulation;
};