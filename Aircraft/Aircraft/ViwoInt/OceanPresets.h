#pragma once

struct OceanParameter;
struct OceanRenderParameter;

class OceanSimulatorPresets
{
public:
	static OceanParameter Mid8000();
	static OceanParameter NvidiaDefault();
	static OceanParameter Invalid();
};

class OceanRendererPresets
{
public:
	static OceanRenderParameter NvidiaDefault();
	static OceanRenderParameter DemoPreset();
	static OceanRenderParameter Invalid();
};
