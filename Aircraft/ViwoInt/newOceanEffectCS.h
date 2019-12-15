#ifndef _NEWOCEANEFFECTCS_H
#define _NEWOCEANEFFECTCS_H

#include <map>
#include <string>
#include <vector>

#include "Effect.h"

#include "VPMath.h"

#include "OceanRefCoord.h"
#include "FluidRigidInteraction/SWE/OceanPatch.h"
#include "FluidRigidInteraction/SWE/RigidBody/Coupling.h"

class OceanGlobalRenderer;
struct OceanParameter;
namespace EffectPlug {
	class DynamicSkyEffect;

	struct BoatParam {
		float m_inertia_scale = 1e9;
		glm::vec3 m_initial_pos = glm::vec3(128,5,128);
		float m_mass = 1e6;
		float m_damping = 0.9;
		float m_patch_length = 512;
		std::string boat_sample_file;
		std::string boat_model_file;
	};

	class OceanEffectCS2 :public Effect
	{
	public:
		typedef int QueryID;

		explicit OceanEffectCS2(DynamicSkyEffect *sky_effect);
		~OceanEffectCS2();
		bool checkEnv();
		bool initialize();
		bool release();

		virtual void Render(void *pOptions = 0, double _time = 0.0, float _deltatime = 0.0f);

		/// interface 
		bool SetOceanParameter(const OceanParameter &ocean_parameter);
		bool GetOceanParameter(OceanParameter &ocean_parameter);

		OceanRefCoord const& GetLocalRef() const;

		QueryID AddNewDisplacementQuery(std::vector<VPE::Basic::CDVector3> const& positions);
		void UpdateDisplacementQuery(QueryID id, std::vector<VPE::Basic::CDVector3> const& positions);
		void RemoveDisplacementQuery(QueryID id);
		std::vector<VPE::Basic::CDVector3> const& GetDisplacementQuery(QueryID id) const;
		std::vector<float> const& GetDisplacementQueryResult(QueryID id) const;


	private:
		OceanPatch *m_ocean_patch = nullptr;

		std::vector<Coupling*> boats;
		
		std::vector<BoatParam> boats_param;

		OceanGlobalRenderer *ocean_entity;

		DynamicSkyEffect *sky_effect;

		bool wire_mode;

		bool enabled;

		int m_boat_num = 0;

		QueryID next_query_id;
		std::map<QueryID, std::vector<VPE::Basic::CDVector3>> queries;
		std::map<QueryID, std::vector<float>> results;
	};
}

#endif
