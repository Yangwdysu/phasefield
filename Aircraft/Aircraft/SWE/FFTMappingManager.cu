#include "FFTMappingManager.h"
#include "gl_utilities.h"
#include "types.h"

FFTMappingManager::FFTMappingManager() {
	
}

FFTMappingManager::~FFTMappingManager() {
	for (int i = 0; i < m_num_fft; ++i) {
		delete m_fft_data[i];
		m_fft_data[i] = nullptr;
	}
	cudaCheck(cudaGraphicsUnregisterResource(m_cuda_displacement_tex));
	cudaCheck(cudaGraphicsUnregisterResource(m_cuda_displacement_tex));
	glDeleteTextures(1, &m_displacement_tex_array);
	glDeleteTextures(1, &m_gradient_tex_array);
}

void FFTMappingManager::initialize() {
	for (int i = 0; i < MAX_FFT_CASCADE; ++i)
	{
		if (i >= m_num_fft) 
		{
			m_fft_data[i] = nullptr;
			continue;
		}
		auto wind_level = m_fft_wind_level[i];
		m_fft_data[i] = new OceanPatch(m_resolution, m_fft_wind_dir[i], m_wind_param[wind_level].windSpeed, m_wind_param[wind_level].A, m_wind_param[wind_level].choppiness, m_wind_param[wind_level].global);
		m_fft_data[i]->m_fft_real_length = m_fft_physical_length[0];
	}
	gl_utility::createTexture2DArray(m_displacement_tex_array, m_resolution, m_resolution, m_num_fft, GL_RGBA32F, GL_REPEAT, GL_LINEAR);
	gl_utility::createTexture2DArray(m_gradient_tex_array, m_resolution, m_resolution, m_num_fft, GL_RGBA32F, GL_REPEAT, GL_LINEAR);
	cudaCheck(cudaGraphicsGLRegisterImage(&m_cuda_displacement_tex, m_displacement_tex_array, GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsWriteDiscard));
	cudaCheck(cudaGraphicsGLRegisterImage(&m_cuda_gradient_tex, m_gradient_tex_array, GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsWriteDiscard));
}

void FFTMappingManager::animate(float time) {
	for (int i = 0; i < m_num_fft; ++i)
	{
		m_fft_data[i]->animate(time*m_fft_flow_speed[i]);
	}
	synchronize();
}

void FFTMappingManager::synchronize() {
	cudaCheck(cudaGraphicsMapResources(1, &m_cuda_displacement_tex));
	cudaCheck(cudaGraphicsMapResources(1, &m_cuda_gradient_tex));
	cudaArray_t cuda_displacement_array = nullptr;
	cudaArray* cuda_gradient_array = nullptr;

	for (unsigned int i = 0; i < m_num_fft; ++i)
	{
		synchronCheck;
		cudaCheck(cudaGraphicsSubResourceGetMappedArray(&cuda_displacement_array, m_cuda_displacement_tex, i, 0));
		cudaCheck(cudaMemcpyToArray(cuda_displacement_array, 0, 0, m_fft_data[i]->m_displacement, m_resolution*m_resolution * sizeof(float4), cudaMemcpyDeviceToDevice));
		cuda_displacement_array = nullptr;

		cudaCheck(cudaGraphicsSubResourceGetMappedArray(&cuda_gradient_array, m_cuda_gradient_tex, i, 0));
		cudaCheck(cudaMemcpyToArray(cuda_gradient_array, 0, 0, m_fft_data[i]->m_gradient, m_resolution*m_resolution * sizeof(float4), cudaMemcpyDeviceToDevice));
		cuda_gradient_array = nullptr;
	}

	cudaCheck(cudaGraphicsUnmapResources(1, &m_cuda_displacement_tex));
	cudaCheck(cudaGraphicsUnmapResources(1, &m_cuda_gradient_tex));
}
