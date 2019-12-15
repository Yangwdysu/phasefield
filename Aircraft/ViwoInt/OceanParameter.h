#pragma once

#include "OceanVec.h"

struct OceanParameter
{
	// Must be power of 2.
	int dmap_dim;
	// Typical value is 1000 ~ 2000 (centimeters)
	float patch_length;

	// Adjust the time interval for simulation.
	float time_scale;
	// Amplitude for transverse wave. Around 1.0
	float wave_amplitude;
	// Wind direction. Normalization not required.
	OceanVec::vec2 wind_dir;
	// Around 100 ~ 1000
	float wind_speed;
	// This value damps out the waves against the wind direction.
	// Smaller value means higher wind dependency.
	float wind_dependency;
	// The amplitude for longitudinal wave. Must be positive.
	float choppy_scale;

	friend class OceanSimulatorPresets;

private:
	OceanParameter(){}
};

struct OceanRenderParameter
{
	// Shading properties:
	// Two colors for waterbody and sky color
	OceanVec::vec3 SkyColor;
	OceanVec::vec3 WaterbodyColor;
	// Blending term for sky cubemap
	float SkyBlending;

	// Perlin wave parameters
	float PerlinSize;
	OceanVec::vec3 PerlinAmplitude;
	OceanVec::vec3 PerlinGradient;
	OceanVec::vec3 PerlinOctave;

	OceanVec::vec3 BendParam;

	// Sunspot parameters
	OceanVec::vec3 SunDir;
	OceanVec::vec3 SunColor;
	float Shineness;

	// Ocean floor parameters
	float OceanDepth;
	float OceanFloorTexSize;  // Width and height in meters

	friend class OceanRendererPresets;

private:
	OceanRenderParameter(){}
};
