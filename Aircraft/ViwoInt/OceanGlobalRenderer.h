#pragma once

#include "GLObjects.h"
#include "GLProgramObject.h"

#include "OceanParameter.h"
#include "OceanRefCoord.h"
#include "OceanReflectContext.h"
#include "OceanVec.h"

#include <GLBufferVector.h>
#include <GLDynamicSizedBuffer.h>
#include <GLSyncPoint.h>
#include <vector>
#include "FluidRigidInteraction/SWE/OceanPatch.h"
#include "FluidRigidInteraction/SWE/RigidBody/Coupling.h"
using std::vector;

namespace EffectPlug
{
	class DynamicSkyEffect;
}

class OceanGlobalRenderer
{
public:

	struct Patch
	{
		OceanVec::dvec2 lonlat[4];
		unsigned neighbourSplit[4];
	};

	struct BatchedPatchVertex
	{
		OceanVec::dvec2 lonlat;
		unsigned neighbourSplit;
		unsigned padding[3];
	};
	static_assert(sizeof(BatchedPatchVertex) == 32, "Attribute alignment incorrect.");

	struct BatchedPatch
	{
		BatchedPatchVertex vertices[4];
	};

	typedef int QueryID;

	OceanGlobalRenderer(
		OceanPatch* fft_patch,
		vector<Coupling*>* boats_vec,
		OceanRenderParameter const& render_param,
		EffectPlug::DynamicSkyEffect const* sky_effect);

	~OceanGlobalRenderer();
	/// reallocate buffer when change window size
	void AllocBuffers(int width, int height);

	void startRendering(OceanVec::dvec3 const& eye, OceanVec::dvec3 const& dir, OceanVec::dvec3 const& up, OceanVec::dmat4 const& projMat, float time);

	OceanRefCoord const& GetLocalRef() const;

	OceanVec::dmat4 StartReflectionRendering();

	void FinishReflectionRendering();

	void startPatchRendering(GLenum mode, bool batched = false);
	void sampleGlobalPositions(std::map<QueryID, std::vector<OceanVec::dvec3>> const& queries);
	void readPreviousSampledPositions(std::map<QueryID, std::vector<float>> &above_sea_level);
	void renderPatch(Patch const& patch);
	void renderPatches(std::vector<BatchedPatch> const& patches);
	void endPatchRendering();

	void DrawUnderwaterMask();

	template <typename Func>
	void modifyRenderParam(Func const& func)
	{
		func(render_param);
		applyStableParam();
	}

	void ReloadShaders();

private:
	void applyStableParam();
	void loadTextures();
	void bindAllTextures();
	void allocateVAO();

	OceanRenderParameter render_param;

	OceanPatch *m_fft_patch;
	vector<Coupling*>* m_boats_vec;

	EffectPlug::DynamicSkyEffect const* sky_effect;

	OceanRefCoord m_ref_coord;

	GLProgramObject m_Program, m_WireProgram, m_MaskProgram;
	GLBufferObject m_TransformInfoBuffer;
	GLBufferObject m_DynamicParamBuffer;
	GLBufferObject m_StableParamBuffer;

	GLVertexArrayObject m_PatchesVAO;
	OpenGLUtils::GLDynamicSizedBuffer<BatchedPatch> m_PatchesBuffer;

	static int const SAMPLING_RESULTS_QUEUE_SIZE = 3;
	GLProgramObject point_sampling_program;
	GLVertexArrayObject sampling_positions_vao;

	int next_sampling_results_slot;
	struct SamplingResults
	{
		std::map<QueryID, std::pair<int, int>> result_ranges;
		OpenGLUtils::GLWriteOnlyVectorT<OceanVec::dvec3>::type sampling_positions;
		OpenGLUtils::GLReadOnlyVectorT<GLfloat>::type results;
		OpenGLUtils::GLSyncPoint sync_point;
	} tasks[SAMPLING_RESULTS_QUEUE_SIZE];

	GLTextureObject m_TexPerlin;
	GLTextureObject m_TexFresnel;
	GLTextureObject m_TexFoam;

	int m_fbo_width, m_fbo_height;
	OceanReflectContext m_reflect_context;

	GLSamplerObject const& m_linear_sampler;
	GLSamplerObject const& m_nearest_sampler;

	bool in_imm_mode;
	int boat_move_tmp = 0;
};
