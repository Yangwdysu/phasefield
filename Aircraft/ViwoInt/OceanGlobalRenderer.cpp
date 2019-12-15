#include <iostream>
#include <sstream>

#include <RenderEngine/IManagedFramebuffer.h>
#include <RenderEngine/Profiler.h>
#include <RenderSystemInterface.h>
#include <ViWoRoot.h>

#include "OceanGlobalRenderer.h"

#include <CoordUtils/Coordinates.h>
#include "CoordUtils/LocalCoord.h"

#include "DynamicSkyCube/DynamicSkyEffect.h"

#include <Passes/GeometryPass.h>

#include "passes/OceanReflectionPass.h"

#include <GLBufferUtils.h>
#include <GLResourceStack.h>
#include <GLSamplerManager.h>
#include <GLShaderUtils.h>
#include <GLTextureUtils.h>
#include <GLVertexArrayObjectGenerator.h>

#include "glenv.h"

#include <FreeImagePlus.h>

#include "ViwoUtils.h"
#include "StaticRenderModel.h"
#include "RenderTree.h"
//#define OCEAN_FACE_DIR_DEBUG
//#define OCEAN_FLOOR_HEIGHT_DEBUG

//#define OCEAN_PROFILING

//#define OCEAN_UNDERWATER_ENABLED
//#define OCEAN_REFLECTION
using namespace OceanVec;

struct TransformInfo
{
	dmat4 projViewMat;
	mat4  projInverse;
	mat4  local2eye;
	mat4  eye2local;
	dvec3 east;
	double unused0;
	dvec3 north;
	double unused1;
	dvec3 surface_up;
	double EarthRadius;
	float baseSplit;
	float unused2;
	unsigned screenWidth, screenHeight;
	vec4 east_clipCoord, north_clipCoord;

	//add by leo xu ocean effect with boat
	dvec2 boat1; // lon lat

	dvec2 boat1_offset; // meter

	dvec2 boat2;

	dvec2 boat2_offset;

	dvec2 boat3;

	dvec2 boat3_offset;

	dvec2 boat4;

	dvec2 boat4_offset;

};

struct DynamicParam
{
	vec2 g_UVBase;
	vec2 g_PerlinMovement;
	vec3 g_LocalEye;
	GLfloat g_choppiness;
	GLfloat g_Time;
	vec3 unused;
};

struct StableParam
{
	// Water-reflected sky color
	vec3		g_SkyColor;
	GLfloat		unused0;
	// The color of bottomless water body
	vec3		g_WaterbodyColor;

	// The strength, direction and color of sun streak
	GLfloat		g_Shineness;
	vec3		g_SunDir;
	GLfloat		unused1;
	vec3		g_SunColor;
	GLfloat		unused2;

	// The parameter is used for fixing an artifact
	vec3		g_BendParam;

	// Perlin noise for distant wave crest
	GLfloat		g_PerlinSize;
	vec3		g_PerlinAmplitude;
	GLfloat		unused3;
	vec3		g_PerlinOctave;
	GLfloat		unused4;
	vec3		g_PerlinGradient;

	// Constants for calculating texcoord from position
	GLfloat		g_TexelLength_x2;
	GLfloat     g_fft_scale;
	GLfloat		g_fft_resolution;
	GLfloat		g_fft_real_length;		// unit meter
	GLfloat		g_capillary_real_length; // unit meter
	GLfloat		g_OceanDepth;         // This parameter only affects underwater surface reflection for the moment.
	GLfloat		g_OceanFloorTexSize;  // In centimeters like other lengths
};

OceanGlobalRenderer::OceanGlobalRenderer(OceanPatch *fft_patch,vector<Coupling*>* boats_vec, OceanRenderParameter const& render_param, EffectPlug::DynamicSkyEffect const* sky_effect)
	: render_param(render_param)
	, m_fft_patch(fft_patch)
	, m_boats_vec(boats_vec)
	, sky_effect(sky_effect)
	, next_sampling_results_slot(0)
	, m_fbo_width(0)
	, m_fbo_height(0)
	, m_linear_sampler(GLSamplerManager::GetLinearSampler())
	, m_nearest_sampler(GLSamplerManager::GetNearestSampler())
	, in_imm_mode(false)
{
	ReloadShaders();

	allocUniformBuffer<TransformInfo>(m_Program, m_TransformInfoBuffer, "TransformInfo");
	allocUniformBuffer<DynamicParam>(m_Program, m_DynamicParamBuffer, "DynamicParam");
	allocUniformBuffer<StableParam>(m_Program, m_StableParamBuffer, "StableParam");

	applyStableParam();
	loadTextures();
	allocateVAO();
#ifdef OCEAN_REFLECTION
	EffectPlug::OceanReflectionPass::GetInstanceRef().Enable();
#endif
}

OceanGlobalRenderer::~OceanGlobalRenderer()
{
#ifdef OCEAN_REFLECTION
	EffectPlug::OceanReflectionPass::GetInstanceRef().Disable();
#endif
}
void OceanGlobalRenderer::AllocBuffers(int width, int height)
{
	if (this->m_fbo_width == width && this->m_fbo_height == height)
		return;

	m_fbo_width = width;
	m_fbo_height = height;
}

void OceanGlobalRenderer::startRendering(OceanVec::dvec3 const& eye, OceanVec::dvec3 const& dir, OceanVec::dvec3 const& up, OceanVec::dmat4 const& projMat, float time)
{
	m_ref_coord.MoveFocus(eye);

	TransformInfo transformInfo;

	auto const& world2eye = OceanVec::lookAtDir(eye, dir, up);

	transformInfo.projViewMat = projMat * world2eye;
	transformInfo.projInverse = OceanVec::inverse(projMat);

	auto const& center = m_ref_coord.GetCenter();
	auto const& e = transformInfo.east = m_ref_coord.GetEast();
	auto const& n = transformInfo.north = m_ref_coord.GetNorth();
	auto const& r = transformInfo.surface_up = m_ref_coord.GetUp();

	OceanVec::dmat4 local2world(e.x, e.y, e.z, 0.0, n.x, n.y, n.z, 0.0, r.x, r.y, r.z, 0.0, center.x, center.y, center.z, 1.0);
	OceanVec::dmat4 const local2eye = world2eye * local2world;
	transformInfo.local2eye = local2eye;
	transformInfo.eye2local = OceanVec::inverse(local2eye);

	transformInfo.EarthRadius = ::EarthRadius;
	transformInfo.baseSplit = 32.f;

	transformInfo.screenWidth = m_fbo_width;
	transformInfo.screenHeight = m_fbo_height;

	transformInfo.east_clipCoord = mul_dmat3x4_dvec3(transformInfo.projViewMat, transformInfo.east);
	transformInfo.north_clipCoord = mul_dmat3x4_dvec3(transformInfo.projViewMat, transformInfo.north);

	//put initial boat position
	//boat origin local coordinate
	
	//auto render_tree = StaticRenderer::RenderTree::GetInstance();
	//auto hangmu_model = render_tree->_static_models[0];
	//auto hangmu_mat = hangmu_model->_modelmat[0];
	//double world_pos[3] = { hangmu_mat[0][3] ,hangmu_mat[1][3],hangmu_mat[2][3]};
	//double lonlatheight[3] = { 0,0,0 };
	//g_coord.GlobalCoord2LongLat(world_pos, lonlatheight);
	//printf("hang mu (lon:%f, lat:%f)\n", lonlatheight[0], lonlatheight[1]);
	//transformInfo.boat1 = dvec2(lonlatheight[0], lonlatheight[1]) / 180.0*M_PI;
	
	transformInfo.boat1 = dvec2(120.136168-0.000001*(boat_move_tmp++),35.767108) / 180.0*M_PI;
	float2 local_boat_center = (*m_boats_vec)[0]->getLocalBoatCenter();
	transformInfo.boat1_offset = dvec2(local_boat_center.x, local_boat_center.y);

	setBufferSubData(GL_UNIFORM_BUFFER, m_TransformInfoBuffer, transformInfo);

	DynamicParam dynamic_param;

	dynamic_param.g_UVBase = vec2(0.0f, 0.0f);

	dynamic_param.g_choppiness = m_fft_patch->getChoppiness();
	auto win_direction = make_float2(cosf(m_fft_patch->windDir), sinf(m_fft_patch->windDir));
	dynamic_param.g_PerlinMovement = glm::vec2(win_direction.x, win_direction.y) * time * m_fft_patch->m_fft_flow_speed*0.01f;

	dvec3 const eyeToRefWorld = eye - center;
	dynamic_param.g_LocalEye = dvec3(dot(eyeToRefWorld, m_ref_coord.GetEast()), dot(eyeToRefWorld, m_ref_coord.GetNorth()), dot(eyeToRefWorld, m_ref_coord.GetUp()));
	setBufferSubData(GL_UNIFORM_BUFFER, m_DynamicParamBuffer, dynamic_param);

	m_reflect_context.makeMatrices(eye, dir, up, m_ref_coord.GetCenter(), m_ref_coord.GetUp(), projMat);
}

OceanRefCoord const& OceanGlobalRenderer::GetLocalRef() const
{
	return m_ref_coord;
}

OceanVec::dmat4 OceanGlobalRenderer::StartReflectionRendering()
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadMatrixd(OceanVec::value_ptr(m_reflect_context.getModelviewMatrix()));
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadMatrixd(OceanVec::value_ptr(m_reflect_context.getProjectionMatrix()));

	// The internal FBO shall have the same size as screen framebuffer.
	// No viewport resetting is needed here.
	ViwoUtils::PushDrawFrambuffer(m_reflect_context.getFBO());
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	return m_reflect_context.getProjectionMatrix() * m_reflect_context.getModelviewMatrix();
}

void OceanGlobalRenderer::FinishReflectionRendering()
{
	ViwoUtils::PopDrawFramebuffer();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	gl_Check();
}

void OceanGlobalRenderer::startPatchRendering(GLenum mode, bool batched)
{
#ifdef OCEAN_UNDERWATER_ENABLED
	gl_Call(glPushAttrib(GL_STENCIL_BUFFER_BIT));
#endif

	if (mode == GL_FILL)
	{
		OpenGLUtils::PushProgram(m_Program);
		gl_Call(glEnable(GL_DEPTH_TEST));
	}
	else
	{
		OpenGLUtils::PushProgram(m_WireProgram);
		gl_Call(glEnable(GL_DEPTH_TEST));
	}

	gl_Call(glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_TransformInfoBuffer.get()));
	gl_Call(glBindBufferBase(GL_UNIFORM_BUFFER, 1, m_DynamicParamBuffer.get()));
	gl_Call(glBindBufferBase(GL_UNIFORM_BUFFER, 2, m_StableParamBuffer.get()));

	bindAllTextures();

#ifdef OCEAN_UNDERWATER_ENABLED
	gl_Call(glClearStencil(0));
	gl_Call(glStencilMask(3));
	gl_Call(glClear(GL_STENCIL_BUFFER_BIT));
	gl_Call(glEnable(GL_STENCIL_TEST));
	gl_Call(glStencilFuncSeparate(GL_FRONT, GL_ALWAYS, 1, 3));
	gl_Call(glStencilFuncSeparate(GL_BACK, GL_ALWAYS, 2, 3));
	gl_Call(glStencilOpSeparate(GL_FRONT_AND_BACK, GL_KEEP, GL_KEEP, GL_REPLACE));
#else
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
#endif

	gl_Call(glEnable(GL_FRAMEBUFFER_SRGB));
	gl_Call(glPatchParameteri(GL_PATCH_VERTICES, 4));
	gl_Call(glPolygonMode(GL_FRONT_AND_BACK, mode));
	if (!batched)
	{
		glBegin(GL_PATCHES);
		in_imm_mode = true;
	}
	else
		in_imm_mode = false;
}

void OceanGlobalRenderer::sampleGlobalPositions(std::map<QueryID, std::vector<OceanVec::dvec3>> const& queries)
{
	SamplingResults &task = tasks[next_sampling_results_slot];
	if (!task.results.empty())
	{
		task.sync_point.wait();
	}
	task.sampling_positions.clear();
	task.result_ranges.clear();

	std::size_t total_num_samples = 0;
	for (auto it = queries.cbegin(); it != queries.cend(); ++it)
	{
		total_num_samples += it->second.size();
	}
	task.sampling_positions.reserve(total_num_samples);

	for (auto it = queries.cbegin(); it != queries.cend(); ++it)
	{
		auto const& query = *it;
		QueryID const id = query.first;
		task.result_ranges[id].first = task.sampling_positions.size();
		task.sampling_positions.insert(task.sampling_positions.end(), query.second.cbegin(), query.second.cend());
		task.result_ranges[id].second = task.sampling_positions.size();
	}

	assert(task.sampling_positions.size() == total_num_samples);

	task.results.clear();
	task.results.resize(total_num_samples);

	glVertexArrayVertexBuffer(sampling_positions_vao.get(), 0, vectorToBuffer(task.sampling_positions), 0, sizeof(OceanVec::dvec3));
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vectorToBuffer(task.results));
	glBindVertexArray(sampling_positions_vao.get());
	glEnable(GL_RASTERIZER_DISCARD);

	OpenGLUtils::PushProgram(point_sampling_program);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glDrawArrays(GL_POINTS, 0, total_num_samples);
	task.sync_point = OpenGLUtils::GLSyncPoint::create();
	OpenGLUtils::PopProgram();

	glDisable(GL_RASTERIZER_DISCARD);
	glBindVertexArray(0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
	glVertexArrayVertexBuffer(sampling_positions_vao.get(), 0, 0, 0, 0);

	next_sampling_results_slot = (next_sampling_results_slot + 1) % SAMPLING_RESULTS_QUEUE_SIZE;
}

void OceanGlobalRenderer::readPreviousSampledPositions(std::map<QueryID, std::vector<float>> &above_sea_level)
{
	SamplingResults &task = tasks[next_sampling_results_slot];

	std::map<QueryID, std::pair<int, int>> const& result_ranges = task.result_ranges;
	OpenGLUtils::GLReadOnlyVectorT<GLfloat>::type const& linear_results = task.results;

	if (!linear_results.empty())
	{
		task.sync_point.wait();
	}

	for (auto it = above_sea_level.begin(); it != above_sea_level.end(); ++it)
	{
		QueryID const id = it->first;
		std::vector<float> &result_out = it->second;
		try
		{
			std::pair<int, int> const& range = result_ranges.at(id);
			result_out.assign(linear_results.cbegin() + range.first, linear_results.cbegin() + range.second);
		}
		catch (...)
		{
			result_out.clear();
		}
	}
}

void OceanGlobalRenderer::renderPatch(Patch const& patch)
{
	for (int i = 0; i < 4; ++i)
	{
		glVertexAttribI1ui(2, patch.neighbourSplit[i]);
		glVertexAttribL2d(0, patch.lonlat[i].x, patch.lonlat[i].y);
	}
}

void OceanGlobalRenderer::renderPatches(std::vector<BatchedPatch> const& patches)
{
	if (patches.size() > 0) {
		BatchedPatch batchpach = patches[0];
		cout << sinf(3.1415926 / 180 * (batchpach.vertices[0].lonlat.x - batchpach.vertices[1].lonlat.x))*::EarthRadius << endl;
	}

	m_PatchesBuffer.Assign(patches);
	glBindVertexArray(m_PatchesVAO.get());
	glBindVertexBuffer(0, m_PatchesBuffer.Get(), 0, sizeof(BatchedPatchVertex));
	glDrawArrays(GL_PATCHES, 0, GLsizei(patches.size()) * 4);
	glBindVertexBuffer(0, 0, 0, 16);
	glBindVertexArray(0);
}

void OceanGlobalRenderer::endPatchRendering()
{
	if (in_imm_mode)
	{
		glEnd();
		in_imm_mode = false;
	}
	gl_Call(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
	gl_Call(glDisable(GL_FRAMEBUFFER_SRGB));

	glDisable(GL_CULL_FACE);

	gl_Call(glBindTextures(0, 18, nullptr));
	gl_Call(glBindSamplers(0, 18, nullptr));
	gl_Call(glBindBuffersBase(GL_UNIFORM_BUFFER, 0, 3, nullptr));

	OpenGLUtils::PopProgram();

#if 0
	gl_Call(glBindTexture(GL_TEXTURE_2D, m_face_stencil_tex.get()));
	gl_Call(glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_fbo_width, m_fbo_height));
#ifdef OCEAN_FACE_DIR_DEBUG
	fipImage debug_image(FIT_BITMAP, m_fbo_width, m_fbo_height, 8);
	gl_Call(glGetTexImage(GL_TEXTURE_2D, 0, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, debug_image.accessPixels()));
	debug_image.save("ocean_dir.tiff", TIFF_NONE);
#endif
	gl_Call(glBindTexture(GL_TEXTURE_2D, 0));
#endif

#ifdef OCEAN_UNDERWATER_ENABLED
	gl_Call(glStencilMask(~GLuint(0)));
	gl_Call(glClear(GL_STENCIL_BUFFER_BIT));

	gl_Call(glPopAttrib());
#endif
}

#ifdef OCEAN_UNDERWATER_ENABLED
void OceanGlobalRenderer::DrawUnderwaterMask()
{
	gl_Call(glPushAttrib(GL_DEPTH_BUFFER_BIT));
	gl_Call(glDisable(GL_DEPTH_TEST));

	gl_Call(glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_TransformInfoBuffer.get()));
	gl_Call(glBindBufferBase(GL_UNIFORM_BUFFER, 1, m_DynamicParamBuffer.get()));
	gl_Call(glBindBuffer(GL_UNIFORM_BUFFER, 0));

	GLuint const textures[3] =
	{ ViWoROOT::GetRenderSystem()->GetCurrentRender()->GetManagedFramebuffer()->GetDrawColorBuffer()
		, ViWoROOT::GetRenderSystem()->GetCurrentRender()->GetManagedFramebuffer()->GetDrawDepthStencilBuffer()
		, ViWoROOT::GetRenderSystem()->GetCurrentRender()->GetManagedFramebuffer()->GetDrawStencilView()
	};

	GLuint const samplers[3] =
	{ m_nearest_sampler.get()
		, m_nearest_sampler.get()
		, 0
	};

	glBindTextures(0, 3, textures);
	glBindSamplers(0, 3, samplers);

	OpenGLUtils::PushProgram(m_MaskProgram);
	glBegin(GL_QUADS);
	glVertexAttrib2f(0, -1.f, -1.f);
	glVertexAttrib2f(0, +1.f, -1.f);
	glVertexAttrib2f(0, +1.f, +1.f);
	glVertexAttrib2f(0, -1.f, +1.f);
	gl_Call(glEnd());
	OpenGLUtils::PopProgram();

	gl_Call(glBindTextures(0, 3, nullptr));
	gl_Call(glBindSamplers(0, 3, nullptr));
	gl_Call(glBindBuffersBase(GL_UNIFORM_BUFFER, 0, 2, nullptr));
	gl_Call(glActiveTexture(GL_TEXTURE0));

	gl_Call(glPopAttrib());
}
#endif

void OceanGlobalRenderer::ReloadShaders()
{
	m_Program = OpenGLUtils::CompileAndLinkShaderProgram(OpenGLUtils::ShaderConfig()
		.Vertex("ocean_global_render.glsl")
		.TessCtrl("ocean_global_render.glsl")
		.TessEval("ocean_global_render.glsl")
		.Fragment("ocean_global_render.glsl")
#ifndef OCEAN_REFLECTION
		.Macro("NO_REFLECTION", "")
#endif
	);
	m_WireProgram = OpenGLUtils::CompileAndLinkShaderProgram(OpenGLUtils::ShaderConfig()
		.Vertex("ocean_global_render.glsl")
		.TessCtrl("ocean_global_render.glsl")
		.TessEval("ocean_global_render.glsl")
		.Fragment("ocean_global_render.glsl")
		.Macro("OCEAN_WIREFRAME", "")
	);
	m_MaskProgram = compileTraditionalShaderFromFile("ocean_underwater_mask.vert", "ocean_underwater_mask.frag");
	point_sampling_program = OpenGLUtils::CompileAndLinkShaderProgram(OpenGLUtils::ShaderConfig()
		.Vertex("ocean_global_sampling.glsl")
		.Fragment("ocean_global_sampling.glsl"));
}

void OceanGlobalRenderer::applyStableParam()
{
	auto fft_patch_length = m_fft_patch->m_fft_real_length_render;
	auto fft_resolution = m_fft_patch->getGridSize();

	StableParam stable_param;
	// Grid side length * 2
	stable_param.g_TexelLength_x2 = fft_patch_length / fft_resolution * 2;
	stable_param.g_fft_resolution = fft_resolution;
	stable_param.g_fft_real_length = fft_patch_length;
	stable_param.g_fft_scale = m_fft_patch->m_windType;
	stable_param.g_capillary_real_length = (*m_boats_vec)[0]->getTrail()->m_patch_length;
	// Color
	stable_param.g_SkyColor = render_param.SkyColor;
	stable_param.g_WaterbodyColor = render_param.WaterbodyColor;

	// Perlin
	stable_param.g_PerlinSize = render_param.PerlinSize;
	stable_param.g_PerlinAmplitude = render_param.PerlinAmplitude;
	stable_param.g_PerlinGradient = render_param.PerlinGradient;
	stable_param.g_PerlinOctave = render_param.PerlinOctave;
	// Multiple reflection workaround
	stable_param.g_BendParam = render_param.BendParam;
	// Sun streaks
	stable_param.g_SunColor = render_param.SunColor;
	stable_param.g_SunDir = render_param.SunDir;
	stable_param.g_Shineness = render_param.Shineness;

	stable_param.g_OceanDepth = render_param.OceanDepth;
	stable_param.g_OceanFloorTexSize = render_param.OceanFloorTexSize;

	setBufferSubData(GL_UNIFORM_BUFFER, m_StableParamBuffer, stable_param);
}

static void setTexParameters(GLenum target, GLint wrapMethod, GLint minFilter, GLint magFilter)
{
	gl_Call(glTexParameteri(target, GL_TEXTURE_WRAP_S, wrapMethod));
	gl_Call(glTexParameteri(target, GL_TEXTURE_WRAP_T, wrapMethod));
	gl_Call(glTexParameteri(target, GL_TEXTURE_WRAP_R, wrapMethod));
	gl_Call(glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter));
	gl_Call(glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter));
}

void OceanGlobalRenderer::loadTextures()
{
	m_TexPerlin.alloc();
	m_TexFresnel.alloc();
	m_TexFoam.alloc();

	fipImage image;
	image.load("textures/perlin_reverted.tiff");
	image.flipVertical();
	gl_Call(glBindTexture(GL_TEXTURE_2D, m_TexPerlin.get()));
	gl_Call(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 64, 64, 0, GL_RGBA, GL_FLOAT, image.accessPixels()));
	setTexParameters(GL_TEXTURE_2D, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
	gl_Call(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 4));
	gl_Call(glGenerateMipmap(GL_TEXTURE_2D));

	image.load("textures/fresnel.tiff");
	gl_Call(glBindTexture(GL_TEXTURE_1D, m_TexFresnel.get()));
	gl_Call(glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.accessPixels()));
	setTexParameters(GL_TEXTURE_1D, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR);

	image.load("textures/water_foam1.png");
	image.flipVertical();
	gl_Call(glBindTexture(GL_TEXTURE_2D, m_TexFoam.get()));
	gl_Call(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, image.accessPixels()));
	setTexParameters(GL_TEXTURE_2D, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
	gl_Call(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 4));
	gl_Call(glGenerateMipmap(GL_TEXTURE_2D));


	gl_Call(glBindTextures(0, 16, nullptr));
}

void OceanGlobalRenderer::bindAllTextures()
{
	VPE::RenderSystem::Profiler *profiler = ViWoROOT::GetIRenderSystem()->GetFrameProfiler();

#ifdef OCEAN_PROFILING
	if (profiler)
		profiler->Tag("Ocean:ResolveMSAA");
#endif

	EffectPlug::GeometryPass &gpass = EffectPlug::GeometryPass::GetInstanceRef();
	VPE::RenderSystem::IManagedFramebuffer *framebuffers = ViWoROOT::GetRenderSystem()->GetCurrentRender()->GetManagedFramebuffer();
	GLuint const buffered_color = framebuffers->GetAuxiColorSS();
	GLuint const buffered_depth = framebuffers->GetAuxiDepthStencilSS();
	gpass.EarlyResolveColorDepthMSAA(buffered_color, buffered_depth);

	int tmpboatid[4] = { 0,0,0,0 };
	for (int i = 0; i < m_boats_vec->size(); ++i)
	{
		tmpboatid[i] = (*m_boats_vec)[i]->getTrail()->getHeightTextureId();
	}

	GLuint const textures[15] =
	{
		m_fft_patch->getDisplacementTextureId()
		, m_TexPerlin.get()
		, m_fft_patch->getGradientTextureId()
		, m_TexFresnel.get()
		, sky_effect->GetSkyCubeTexture()
		, 0
#ifdef OCEAN_REFLECTION
		, EffectPlug::OceanReflectionPass::GetInstanceRef().GetColorTexture()
		, EffectPlug::OceanReflectionPass::GetInstanceRef().GetDepthTexture()
#else
		,0
		,0
#endif
		, buffered_color
		, buffered_depth
		, m_TexFoam.get()
		, tmpboatid[0]
		, tmpboatid[1]
		, tmpboatid[2]
		, tmpboatid[3]
	};

	GLuint const samplers[15] =
	{ 0
		, 0
		, 0
		, 0
		, 0
		, 0
		, m_linear_sampler.get()
		, m_nearest_sampler.get()
		, m_linear_sampler.get()
		, m_nearest_sampler.get()
		, 0
		, 0
		, 0
		, 0
		, 0
	};

	glBindTextures(0, 15, textures);
	glBindSamplers(0, 15, samplers);

#ifdef OCEAN_PROFILING
	if (profiler)
		profiler->Tag("Ocean:Dispatch");
#endif

	gl_Call(glActiveTexture(GL_TEXTURE0));
	gl_Call(glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS));
}

void OceanGlobalRenderer::allocateVAO()
{
	m_PatchesVAO.create();
	glVertexArrayAttribBinding(m_PatchesVAO.get(), 0, 0);
	glVertexArrayAttribBinding(m_PatchesVAO.get(), 2, 0);
	glVertexArrayAttribLFormat(m_PatchesVAO.get(), 0, 2, GL_DOUBLE, offsetof(BatchedPatchVertex, lonlat));
	glVertexArrayAttribIFormat(m_PatchesVAO.get(), 2, 1, GL_UNSIGNED_INT, offsetof(BatchedPatchVertex, neighbourSplit));
	glEnableVertexArrayAttrib(m_PatchesVAO.get(), 0);
	glEnableVertexArrayAttrib(m_PatchesVAO.get(), 2);

	sampling_positions_vao = OpenGLUtils::GLVertexArrayObjectGenerator()
		.Binding<OceanVec::dvec3>()
		.AttributeDouble(3, GL_DOUBLE)
		.Finish();
}
