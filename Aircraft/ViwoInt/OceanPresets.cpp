#include "OceanPresets.h"

#include <type_traits>

#include "OceanParameter.h"

OceanParameter OceanSimulatorPresets::Mid8000()
{
	OceanParameter ocean_param = NvidiaDefault();
	ocean_param.patch_length = 8000.f;
	ocean_param.wave_amplitude *= 0.1875f;
	ocean_param.choppy_scale *= 0.75f;
	return ocean_param;
}

OceanParameter OceanSimulatorPresets::NvidiaDefault()
{
	OceanParameter ocean_param;

	// The size of displacement map. In this sample, it's fixed to 512.
	ocean_param.dmap_dim			= 512;
	// The side length (world space) of square patch, in centimeters.
	ocean_param.patch_length		= 2000.0f;
	// Adjust this parameter to control the simulation speed
	ocean_param.time_scale			= 0.8f;
	// A scale to control the amplitude. Not the world space height
	ocean_param.wave_amplitude		= 0.35f;
	// 2D wind direction. No need to be normalized
	ocean_param.wind_dir			= OceanVec::vec2(0.8f, 0.6f);
	// The bigger the wind speed, the larger scale of wave crest.
	// But the wave scale can be no larger than patch_length
	ocean_param.wind_speed			= 600.0f;
	// Damp out the components opposite to wind direction.
	// The smaller the value, the higher wind dependency
	ocean_param.wind_dependency		= 0.07f;
	// Control the scale of horizontal movement. Higher value creates
	// pointy crests.
	ocean_param.choppy_scale		= 1.3f;

	return ocean_param;
}

OceanParameter OceanSimulatorPresets::Invalid()
{
	static_assert(std::is_trivially_copy_assignable<OceanParameter>::value, "Parameter is not plain data");
	OceanParameter param;
	memset(&param, 0, sizeof(param));
	return param;
}

OceanRenderParameter OceanRendererPresets::NvidiaDefault()
{
	OceanRenderParameter param;

	param.SkyColor= OceanVec::vec3(0.38f, 0.45f, 0.56f);
	param.WaterbodyColor = OceanVec::vec3(0.07f, 0.15f, 0.2f);
	param.SkyBlending = 16.0f;

	param.PerlinSize = 10.0f;
	param.PerlinAmplitude = OceanVec::vec3(35, 42, 57);
	param.PerlinGradient = OceanVec::vec3(1.4f, 1.6f, 2.2f);
	param.PerlinOctave = OceanVec::vec3(1.12f, 0.59f, 0.23f);

	param.BendParam = OceanVec::vec3(0.1f, -0.4f, 0.2f);

	param.SunDir = OceanVec::vec3(0.936016f, -0.343206f, 0.0780013f);
	param.SunColor = OceanVec::vec3(1.0f, 1.0f, 0.6f);
	param.Shineness = 400.0f;

	param.OceanDepth = 5.0f;
	param.OceanFloorTexSize = 20.0f;

	return param;
}

OceanRenderParameter OceanRendererPresets::DemoPreset()
{
	OceanRenderParameter param = NvidiaDefault();
	param.SunDir = OceanVec::normalize(OceanVec::vec3(-128, -10, 114));
	param.Shineness = 3200.f;
	return param;
}

OceanRenderParameter OceanRendererPresets::Invalid()
{
	static_assert(std::is_trivially_copy_assignable<OceanRenderParameter>::value, "Parameter is not plain data.");
	OceanRenderParameter param;
	memset(&param, 0, sizeof(param));
	return param;
}
