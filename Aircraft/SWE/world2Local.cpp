#include "world2Local.h"

double World2Local::s_earth_radius = 6373000.0;

glm::dvec3 World2Local::s_center = glm::dvec3(1, 0, 0);

glm::dvec3 World2Local::s_east = glm::dvec3(1, 0, 0);

glm::dvec3 World2Local::s_north = glm::dvec3(0, 1, 0);

glm::dvec3 World2Local::s_surface_up = glm::dvec3(0, 0, 1);


glm::dmat4 World2Local::s_world2local = glm::dmat4();
glm::dmat4 World2Local::s_local2world = glm::dmat4();
glm::dmat4 World2Local::s_local2simulation = glm::dmat4(1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1);
glm::dmat4 World2Local::s_simulation2local = glm::dmat4(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);
glm::dmat4 World2Local::s_simulation2world = glm::dmat4();
glm::dmat4 World2Local::s_world2simulation = glm::dmat4();


glm::vec2 World2Local::world2LocalSurface(glm::dvec3 p)
{
	const glm::dvec3 bb = glm::cross(s_surface_up, glm::normalize(p));
	// Use linear approximation for points within one degree
	// Do not work for points on the opposite side
	if (glm::length(bb) < 0.0175 && glm::dot(s_surface_up, p) > 0)
	{
		const glm::dvec3 worldDist = p - s_surface_up;
		return glm::dvec2(glm::dot(worldDist, s_east), glm::dot(worldDist, s_north))*s_earth_radius;
	}
	else
	{
		const glm::dvec3 b = glm::normalize(bb);
		const glm::dvec3 t = glm::cross(b, s_surface_up);
		const double sint = glm::dot(cross(s_north, b), s_surface_up);
		const double cost = glm::dot(s_north, b);
		const double cosp = glm::dot(s_surface_up, p);
		const double phi = acos(float(cosp));
		return glm::dvec2(cost * phi, sint * phi)*s_earth_radius;
	}
}