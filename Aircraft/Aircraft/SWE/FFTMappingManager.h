#pragma once
#include "OceanPatch.h"
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#define MAX_FFT_CASCADE 4

class FFTMappingManager {
public:
	FFTMappingManager();
	~FFTMappingManager();

	void initialize();

	void animate(float time); 

	void synchronize();

public:
	// decide shape of fft wave
	int m_resolution = 256;
	int m_num_fft = 1;
	float m_fft_wind_dir[MAX_FFT_CASCADE] = {};
	int m_fft_wind_level[MAX_FFT_CASCADE] = {}; // there are [0,12] kinds of wind level from peace ocean to rough ocean
	float m_fft_flow_speed[MAX_FFT_CASCADE] = {};

	//decide the mapping in the rendering texture
	int m_num_mapping = 1;
	float m_fft_physical_length[MAX_FFT_CASCADE] = {};
	float m_fft_amplitude[MAX_FFT_CASCADE] = {};
	float m_fft_choppiness[MAX_FFT_CASCADE] = {};

	// mapping relationship
	int m_mapping[MAX_FFT_CASCADE] = {};

	//public data
	GLuint m_displacement_tex_array;
	GLuint m_gradient_tex_array;

	cudaGraphicsResource_t m_cuda_displacement_tex;
	cudaGraphicsResource_t m_cuda_gradient_tex;

	OceanPatch* m_fft_data[MAX_FFT_CASCADE] = {};

private:
	WindParam m_wind_param[13] = { 
		{0.0f, 0.0f, 0.0f, 0.0f},
		{0.8f, 1.2e2f, 6.0f, 1.5f},
		{1.6f, 3.2e-5f, 6.0f, 1.5f},
		{3.4f, 3.5e-6f, 3.0f, 1.5f},
		{5.5f, 2.5e-6f, 3.0f, 1.5f},
		{8.0f, 2.5e-6f, 3.0f, 1.5f},
		{10.8f, 2.5e-6f, 3.0f, 1.5f},
		{13.9f, 1.2e-6f, 3.0f, 1.5f},
		{17.2f, 2.4e-6f, 2.34f, 1.5f},
		{20.8f, 3.6e-6f, 1.54f, 1.5f},
		{24.5f, 4.2e-6f, 1.2f, 1.5f},
		{28.5f, 6.0e-6f, 0.87f, 1.5f},
		{32.7f, 1.0e-5f, 0.798f, 1.5f} 
	};
};