
#include "MfdNumericalMethod/SPH/BasicSPH.h"
#include "MfdMath/DataStructure/Grid.h"
#include "MfdSurfaceReconstruction/CIsoSurface.h"
#include "MfdMath/LinearSystem/MeshlessSolver.h"

#include "Eigen/Sparse"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/Core"

namespace mfd {

	typedef Grid<3, Vector3f> GridV3f;
	typedef Grid<3, bool>	Grid3b;

	struct Triple
	{
		int x, y, z;
	};

	struct Coeff
	{
		float a;
		float x0;
		float x1;
		float y0;
		float y1;
		float z0;
		float z1;
	};

#define NUM_THREAD 3

#define DENSITY_THRESHOLD 1e-10
#define DEMO_SEPERATION

#define PHASE_SIZE 1


	class UniformGridQuery;
	class HybridMultiPhaseFluid :
		public BasicSPH
	{
	public:
		HybridMultiPhaseFluid(void);
		~HybridMultiPhaseFluid(void);

		virtual bool Initialize(string in_filename = "NULL");

		void InitialSeparation();

		virtual void StepEuler(float dt);

		// querying neighbors
		void ComputeNeighbors();

		//compute density
		virtual void ComputeDensity();


		int GetAirParticleNumber() { return posGrid_Air.elementCount; }
		void LinearSolve(Grid3f& d, Grid3f& d0, float a, float c);


		template<typename T>
		void AdvectForward(Grid<3, T>& d, Grid<3, T>& d0, GridV3f& v, float dt);
	

		void MarkSolidDomain();

		void SetScalarFieldBoundary(Grid3f& field, bool postive);

		void AllocateMemoery(int _np, int _nx, int _ny, int _nz);

		void Invoke(unsigned char type, unsigned char key, int x, int y);
		//virtual void PostProcessing();




		float CFL();

		template<typename T>
		void UpdatePhi(Grid<3, T>& d, Grid<3, T>& d0, float dt);

	public:
		//Grid3f vel_u;
		//Grid3f vel_v;
		//Grid3f vel_w;

		Grid3f vel_u_boundary;
		Grid3f vel_v_boundary;
		Grid3f vel_w_boundary;

		Grid3f pre_vel_u;
		Grid3f pre_vel_v;
		Grid3f pre_vel_w;
		Grid<3, Coeff> coef_u;
		Grid<3, Coeff> coef_v;
		Grid<3, Coeff> coef_w;


		Eigen::VectorXd x0;
		//LinearSystem sys;
		//volecity and pressure
		//Grid3f vel_u1, vel_v1, vel_w1;
		Grid3f p;
		Grid3f divu;

		Grid3f rhoGrid_Air;		//grid density contributed by the air
		Grid3f preGrid_Air;
		Grid3f volGrid_Air;
		Grid3f preMassGrid_Air;
		Grid3f surfaceEnergyGrid_Air;

		GridV3f preVelGrid_Air;
		GridV3f posGrid_Air;
		GridV3f accGrid_Air;

		Grid3f rhoGrid_Liquid;	//grid density contributed by the liquid
		Grid3f fraction_Liquid;
		Grid3f volGrid_Liquid;


		Grid3b marker_Solid;

		Array<float> rhoAirArr;
		Array<float> phiLiquid;
		Array<float> energyLiquid;

		// 	Grid3f f_catche;
		// 	GridV3f vec_catche;

		Vector3f origin;

		float rhoLiquidRef;


		float massLiquid;

		float correctionFactor;


		float rhoAirRef;
		float massAir;
		Grid3f massGrid_Air;
		GridV3f velGrid_Air;
		Grid3b marker_Air;


		float vis_phase[PHASE_SIZE];
		float rho_phase[PHASE_SIZE];
		float mass_phase[PHASE_SIZE];
		Grid3f massGrid_phase[PHASE_SIZE];
		GridV3f velGrid_phase[PHASE_SIZE];
		Grid3b marker_phase[PHASE_SIZE];
		Grid3f extraDiv[PHASE_SIZE];
		GridV3f realposGrid_Air[PHASE_SIZE];
		GridV3f preposGrid_Air;

		float rho1;
		float rho2;

		float vis1;
		float vis2;

		float diff;

		int Nx;
		int Ny;
		int Nz;

		int n_b;

		float V_grid;

		Array<NeighborList> liquidNeigbhors;
		Array<NeighborList> airNeigbhors;

		UniformGridQuery* m_uniGrid;

		KernelFactory::KernelType densityKern;

		int gridNeighborSize;
		Triple fixed_ids[NEIGHBOR_SIZE];
		float fixed_weights[NEIGHBOR_SIZE];

		float viscosityAir;

		float surfacetension;
		float sharpening;
		float compressibility;
		float velocitycoef;
		float diffuse;

		float overall_incompressibility;

		float seprate_incompressibility;
		float seprate_sharpening;

		float particle_incompressibility;
		float particle_sharpening;

		CIsoSurface<float> mc;
		CIsoSurface<float> mc2;
		Vector3f bunnyori;
		int render_id;

		float* ren_massfield;
		float ren_mass;
		bool* ren_marker;
	};

}
