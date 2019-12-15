#include "newOceanEffectCS.h"

#include "Camera.h"
#include "ViWoRoot.h"
#include "CameraManager.h"
#include "ITerrain.h"
#include "RenderCheck.h"
#include "CameraMover.h"
#include "Coordination.h"
#include "SceneGraph/SceneManagers.h"
#include "utilsOpenGL.h"
#include "RenderSystemInterface.h"
#include "RenderEngine/IManagedFramebuffer.h"
#include "RenderInterface.h"

#include "EffectManager.h"
#include "EffectManagerInterface.h"
#include <EMInterface.h>

#include "glenv.h"
#include "OceanGlobalRenderer.h"
#include "OceanParameter.h"
#include "OceanPresets.h"
#include "glutils.h"

#include <CoordUtils/Coordinates.h>

#include "DynamicSkyCube/DynamicSkyEffect.h"
#include "DynamicSkyCube/GlobalSunStatus.h"
#include "DynamicSkyCube/SunStatus.h"

#include <Passes/GeometryPass.h>

#include <FreeImagePlus.h>
#include <fstream>
#include <string.h>

//#define OCEAN_PROFILING

#ifdef OCEAN_PROFILING
#include <RenderEngine/Profiler.h>
#endif

#include "passes/OceanReflectionPass.h"
#include "FluidRigidInteraction/SWE/RigidBody/RigidBody.h"
#include "FluidRigidInteraction/SWE/CapillaryWave.h"

using namespace std;
using namespace VPE::Basic;
using namespace EffectPlug;

static OceanVec::dvec2 ref_lonlat(120. * (M_PI / 180), 38.5 * (M_PI / 180));

/// test reflection and refraction
static GLTextureObject *gridTexture;

vector<CDVector3> box;
static void draw_box()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glPushMatrix();
	glEnable(GL_DEPTH_TEST);
	glBegin(GL_QUADS);
	for (int i = 0; i < int(box.size()); i++)
	{
		CVector3 temp = box[i];
		glVertex3fv(temp.Get());
	}
	glEnd();
	glPopMatrix();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
void debug_box()
{
	CCamera *camera = ViWoROOT::GetCameraManager()->GetCurrCamera();
	ViWoROOT::GetIRenderSystem()->ApplyCamera(camera);
	draw_box();
}

static void loadTestTexture()
{
	gridTexture = new GLTextureObject;
	gridTexture->alloc();
	fipImage gridImage;
	gridImage.load("textures/grid.png");
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gridTexture->get());
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gridImage.getWidth(), gridImage.getHeight(), 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, gridImage.accessPixels());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8);
	glGenerateMipmap(GL_TEXTURE_2D);
}

template <typename VertexFunc, typename TexCoordFunc>
static void drawBoard(OceanVec::dvec3 const& center, VertexFunc const& vertex, TexCoordFunc const& texCoord, double const size = 1.0, double const rotation = 0.0)
{
	using namespace OceanVec;
	dvec3 const boardCenter = center;
	dvec3 const boardOut = normalize(boardCenter);
	dvec3 const boardRight = normalize(cross(dvec3(0., 0., 1.), boardOut));
	dvec3 const boardFront = normalize(cross(boardOut, boardRight));
	dvec3 const boardUp = boardFront * std::sin(radians(rotation)) + boardOut * std::cos(radians(rotation));
	float const texX = 16, texY = 16;
	glEnable(GL_TEXTURE_2D);
	bindTextureUnit(0, GL_TEXTURE_2D, *gridTexture);
	glBegin(GL_QUADS);
	glColor4f(1.f, 1.f, 1.f, 1.f);
	texCoord(0.f, 0.f);
	vertex(boardCenter + (-boardRight - boardUp) * size);
	texCoord(texX, 0.f);
	vertex(boardCenter + (boardRight - boardUp) * size);
	texCoord(texX, texY);
	vertex(boardCenter + (boardRight + boardUp) * size);
	texCoord(0.f, texY);
	vertex(boardCenter + (-boardRight + boardUp) * size);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

template <typename VertexFunc, typename ColorFunc>
static void drawAxis(
	OceanVec::dvec3 const& center,
	OceanVec::dvec3 const& x,
	OceanVec::dvec3 const& y,
	OceanVec::dvec3 const& z,
	VertexFunc const& vertex,
	ColorFunc const& color,
	double const size = 20.0)
{
	using namespace OceanVec;
	glBegin(GL_LINES);
	color(1.f, 0.f, 0.f, 1.f);
	vertex(center);
	vertex(center + x * size);
	color(0.f, 1.f, 0.f, 1.f);
	vertex(center);
	vertex(center + y * size);
	color(0.f, 0.f, 1.f, 1.f);
	vertex(center);
	vertex(center + z * size);
	glEnd();
}

OceanEffectCS2::OceanEffectCS2(DynamicSkyEffect *sky_effect)
	: wire_mode(false)
	, enabled(true)
	, sky_effect(sky_effect)
{
	fstream filein("../../App.Effect/newOcean/ocean-effect-config.txt");
	if (!filein.is_open()) { cout << "../../App.Effect/newOcean/ocean-effect-config.txt" << endl; }
	string property_name;
	filein >> property_name >> wire_mode;
	filein >> property_name >> m_boat_num;
	for (int i = 0; i < m_boat_num; ++i)
	{
		BoatParam param;
		filein >> property_name >> param.m_inertia_scale;
		filein >> property_name >> param.m_initial_pos.x >> param.m_initial_pos.y >> param.m_initial_pos.z;
		filein >> property_name >> param.m_mass;
		filein >> property_name >> param.m_damping;
		filein >> property_name >> param.m_patch_length;
		filein >> property_name >> param.boat_sample_file;
		filein >> property_name >> param.boat_model_file;
		boats_param.push_back(param);
	}
	filein.close();
}

OceanEffectCS2::~OceanEffectCS2()
{

}

bool OceanEffectCS2::initialize()
{
	ViWoROOT::GetTerrainInterface()->SetOceanRenderMode(true);

	try
	{
		OceanRenderParameter render_param = OceanRendererPresets::DemoPreset();

		m_ocean_patch = new OceanPatch();
		boats.clear();
		for (int i = 0; i < m_boat_num; ++i)
		{
			RigidBody* carrier = new RigidBody();
			carrier->initialize(boats_param[i].boat_model_file, boats_param[i].boat_sample_file);
			carrier->setInertiaTensor(boats_param[i].m_inertia_scale*glm::mat3(10.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f));
			carrier->setCenter(boats_param[i].m_initial_pos);
			carrier->setMass(boats_param[i].m_mass);
			carrier->setVelocityDamping(boats_param[i].m_damping);

			CapillaryWave* trail = new CapillaryWave(m_ocean_patch->getGridSize(), boats_param[i].m_patch_length);
			trail->initialize();

			Coupling* coup = new Coupling(m_ocean_patch);
			coup->initialize(carrier, trail);
			coup->setHeightShift(4.0);
			boats.push_back(coup);
		}

		ocean_entity = new OceanGlobalRenderer(m_ocean_patch, &boats, render_param, sky_effect);
		

		GLint viewport_size[4];
		glGetIntegerv(GL_VIEWPORT, viewport_size);
		ocean_entity->AllocBuffers(viewport_size[2], viewport_size[3]);

		loadTestTexture();

		ViWoROOT::GetRenderSystem()->GetCurrentRender()->EnableManagedFramebuffer();
		GeometryPass::GetInstanceRef().Enable();

		return true;
	}
	catch (...)
	{
		return false;
	}
}

bool OceanEffectCS2::release()
{
	GeometryPass::GetInstanceRef().Disable();
	ViWoROOT::GetRenderSystem()->GetCurrentRender()->DisableManagedFramebuffer();

	if (ocean_entity) {
		delete ocean_entity;
		ocean_entity = nullptr;
	}
	if (m_ocean_patch) {
		delete m_ocean_patch;
		m_ocean_patch = nullptr;
	}
	for (int i = 0; i < m_boat_num; ++i)
	{
		delete boats[i]->getBoat();
		delete boats[i]->getTrail();
		delete boats[i];
	}
	boats.clear();

	ViWoROOT::GetTerrainInterface()->SetOceanRenderMode(false);

	return true;
}

bool OceanEffectCS2::checkEnv()
{
	return GLEW_VERSION_4_5;
}

static int DrawTerrainPatches(
	OceanGlobalRenderer *ocean_entity,
	std::vector<double> const& patch_lonlat,
	std::vector<int> const& patch_split)
{
	using namespace OceanVec;

	double const DEG_TO_RAD = M_PI / 180.0;

	int patchCount = 0;
	for (std::size_t i = 0; i < patch_lonlat.size(); i += 4)
	{
		double const minx = patch_lonlat[i] * DEG_TO_RAD;
		double const miny = patch_lonlat[i + 1] * DEG_TO_RAD;
		double const maxx = patch_lonlat[i + 2] * DEG_TO_RAD;
		double const maxy = patch_lonlat[i + 3] * DEG_TO_RAD;

		OceanGlobalRenderer::Patch patch;

		patch.lonlat[0] = dvec2(minx, miny);
		patch.lonlat[1] = dvec2(maxx, miny);
		patch.lonlat[2] = dvec2(maxx, maxy);
		patch.lonlat[3] = dvec2(minx, maxy);

		patch.neighbourSplit[0] = patch_split[i];
		patch.neighbourSplit[1] = patch_split[i + 1];
		patch.neighbourSplit[2] = patch_split[i + 2];
		patch.neighbourSplit[3] = patch_split[i + 3];

		ocean_entity->renderPatch(patch);
		patchCount += 1;
	}
	return patchCount;
}

static void TranslateTerrainPatches
(std::vector<double> const& patch_lonlat
	, std::vector<int> const& patch_split
	, std::vector<OceanGlobalRenderer::BatchedPatch> &patches
)
{
	using namespace OceanVec;

	double const DEG_TO_RAD = M_PI / 180.0;

	patches.clear();
	for (std::size_t i = 0; i < patch_lonlat.size(); i += 4)
	{
		double const minx = patch_lonlat[i] * DEG_TO_RAD;
		double const miny = patch_lonlat[i + 1] * DEG_TO_RAD;
		double const maxx = patch_lonlat[i + 2] * DEG_TO_RAD;
		double const maxy = patch_lonlat[i + 3] * DEG_TO_RAD;

		OceanGlobalRenderer::BatchedPatch const patch =
		{ {
			{ dvec2(minx, miny), patch_split[i]  ,{ 0, 0, 0 } },
			{ dvec2(maxx, miny), patch_split[i + 1],{ 0, 0, 0 } },
			{ dvec2(maxx, maxy), patch_split[i + 2],{ 0, 0, 0 } },
			{ dvec2(minx, maxy), patch_split[i + 3],{ 0, 0, 0 } },
			} };

		patches.push_back(patch);
	}
}

void OceanEffectCS2::Render(void *pOptions, double _time, float _deltatime)
{
	using namespace OceanVec;

#ifdef OCEAN_PROFILING
	VPE::RenderSystem::Profiler *profiler = ViWoROOT::GetIRenderSystem()->GetFrameProfiler();
	if (profiler)
		profiler->Tag("Ocean:Prepare");
#endif

	/// Ocean Effect
	float time = float(EMInterface::GetModuleInstance()->GetEffectManager()->GetTimer().GetTime());

	SunStatus const& sun_status = GlobalSunStatus::GetInstance().GetSun();
	ocean_entity->modifyRenderParam([&sun_status](OceanRenderParameter &param) {
		param.SunDir = sun_status.GetLocalSunDirection();
	});

	gl_Call(glPushAttrib(GL_DEPTH_BUFFER_BIT));

	gl_Call(glDisable(GL_ALPHA_TEST));

	VPE::RenderSystem::IManagedFramebuffer *framebuffers = ViWoROOT::GetRenderSystem()->GetCurrentRender()->GetManagedFramebuffer();
	int const width = framebuffers->GetBufferWidth();
	int const height = framebuffers->GetBufferHeight();

	ocean_entity->AllocBuffers(width, height);

#ifdef OCEAN_PROFILING
	if (profiler)
		profiler->Tag("Ocean:FetchPatches");
#endif

	static std::vector<double> patch_lonlat;
	static std::vector<int> patch_split;
	patch_lonlat.clear();
	patch_split.clear();
	ViWoROOT::GetTerrainInterface()->GetTreeInfo(patch_lonlat, patch_split);
	assert(patch_lonlat.size() == patch_split.size() && patch_lonlat.size() % 4 == 0);

	static std::vector<OceanGlobalRenderer::BatchedPatch> gathered_patches;
	TranslateTerrainPatches(patch_lonlat, patch_split, gathered_patches);

#ifdef OCEAN_PROFILING
	if (profiler)
		profiler->Tag("Ocean:Update");
#endif

	m_ocean_patch->animate(_time);
	for (int i = 0; i < boats.size(); ++i)
	{
		//interface to control boat
		boats[i]->propel(100.0f);

		boats[i]->animate(min(_deltatime,0.016f));
	}

#ifdef OCEAN_PROFILING
	if (profiler)
		profiler->Tag("Ocean:PreRender");
#endif

	CCamera *camera = ViWoROOT::GetCameraManager()->GetCurrCamera();
	ViWoROOT::GetIRenderSystem()->ApplyCamera(camera);

	OceanVec::dvec3 eye = camera->getPosition();
	OceanVec::dvec3 dir = camera->getDirection();
	OceanVec::dvec3 up = camera->getUp();
	OceanVec::dmat4 projMat = camera->getProjectMatrixd();

	ocean_entity->startRendering(eye, dir, up, projMat, time);


	gl_Call(glEnable(GL_DEPTH_TEST));

	if (enabled)
	{
		ocean_entity->startPatchRendering(GL_FILL, true);

#ifdef OCEAN_PROFILING
		if (profiler)
			profiler->Tag("Ocean:Patches");
#endif
		ocean_entity->renderPatches(gathered_patches);
		ocean_entity->endPatchRendering();
	}

	if (wire_mode)
	{
		ocean_entity->startPatchRendering(GL_LINE, true);
		ocean_entity->renderPatches(gathered_patches);
		ocean_entity->endPatchRendering();
	}

	gl_Call(glPopAttrib());
}
bool OceanEffectCS2::SetOceanParameter(const OceanParameter &ocean_parameter)
{
	//pOceanSimulator_->SetParameter(ocean_parameter);
	return false;
}
bool OceanEffectCS2::GetOceanParameter(OceanParameter &ocean_parameter)
{

	//pOceanSimulator_->GetParameter(ocean_parameter);
	return false;
}

OceanRefCoord const& OceanEffectCS2::GetLocalRef() const
{
	return ocean_entity->GetLocalRef();
}


EffectPlug::OceanEffectCS2::QueryID EffectPlug::OceanEffectCS2::AddNewDisplacementQuery(std::vector<VPE::Basic::CDVector3> const& positions)
{
	QueryID id = next_query_id++;
	queries[id] = positions;
	results[id];
	return id;
}

void EffectPlug::OceanEffectCS2::UpdateDisplacementQuery(QueryID id, std::vector<VPE::Basic::CDVector3> const& positions)
{
	queries[id] = positions;
}

void EffectPlug::OceanEffectCS2::RemoveDisplacementQuery(QueryID id)
{
	queries.erase(id);
	results.erase(id);
}

std::vector<VPE::Basic::CDVector3> const& EffectPlug::OceanEffectCS2::GetDisplacementQuery(QueryID id) const
{
	return queries.at(id);
}

std::vector<float> const& EffectPlug::OceanEffectCS2::GetDisplacementQueryResult(QueryID id) const
{
	return results.at(id);
}
