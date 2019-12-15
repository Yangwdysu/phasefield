layout(std140, binding = 0) uniform TransformInfo
{
    dmat4 projViewMat;
    mat4  projInverse;
    mat4  local2eye;
	mat4  eye2local;
    dvec3 east, north, surface_up;
    double EarthRadius;
    float baseSplit;
    uvec2 screenSize;
    vec4 east_clipCoord, north_clipCoord;
	dvec2 boat1,boat1_offset,boat2,boat2_offset,boat3,boat3_offset,boat4,boat4_offset;
};

layout(std140, binding = 1) uniform DynamicParam
{
    vec2 g_UVBase;
    vec2 g_PerlinMovement;
    vec3 g_LocalEye;
	float g_chopiness;
	float g_Time;
};

layout(std140, binding = 2) uniform StableParam
{
    vec3        g_SkyColor;
    vec3        g_WaterbodyColor;

    float       g_Shineness;
    vec3        g_SunDir;
    vec3        g_SunColor;

    vec3        g_BendParam;

    float       g_PerlinSize;
    vec3        g_PerlinAmplitude;
    vec3        g_PerlinOctave;
    vec3        g_PerlinGradient;

    float       g_TexelLength_x2;
	float		g_fft_scale;
	float       g_fft_resolution;
	float		g_fft_real_length;
	float		g_capillary_real_length;
    float       g_OceanDepth;         // This parameter only affects underwater surface reflection for the moment.
    float       g_OceanFloorTexSize;  // In centimeters like other lengths
};

layout(binding = 0)  uniform sampler2D       g_texDisplacement;       // FFT wave displacement map in VS
layout(binding = 1)  uniform sampler2D       g_texPerlin;             // Perlin wave displacement & gradient map in both VS & PS
layout(binding = 2)  uniform sampler2D       g_texGradient;           // FFT wave gradient map in PS
layout(binding = 3)  uniform sampler1D       g_texFresnel;            // Fresnel factor lookup table
layout(binding = 4)  uniform samplerCube     g_texReflectCube;        // A small skybox cube texture for reflection
//layout(binding = 5)  uniform sampler2D       g_texOceanFloor;         // Ocean floor texture

layout(binding = 6)  uniform sampler2D       g_texReflectColorFlip;       // Reflection color texture
layout(binding = 7)  uniform sampler2D       g_texReflectDepthRawFlip;    // Reflection depth, raw value

layout(binding = 8)  uniform sampler2D       g_texRefractColor;       // Refraction color texture
layout(binding = 9)  uniform sampler2D       g_texRefractDepthRaw;    // Refraction depth, raw value
layout(binding = 10) uniform sampler2D		 g_texFoam;				//Foam texture;
layout(binding = 11) uniform sampler2D       g_texBoat1;           
layout(binding = 12) uniform sampler2D       g_texBoat2;
layout(binding = 13) uniform sampler2D       g_texBoat3;
layout(binding = 14) uniform sampler2D       g_texBoat4;

const float PATCH_BLEND_BEGIN = 8;
const float PATCH_BLEND_END = 600;
const float WATER_REFRACTION = 1.33;  // Be aware that the fresnel reflectivity and trasmissivity relies on the refraction index.
