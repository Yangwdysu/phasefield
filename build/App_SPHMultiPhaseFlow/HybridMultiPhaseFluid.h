#pragma once
#include "MfdNumericalMethod/SPH/BasicSPH.h"
#include "MfdMath/DataStructure/Grid.h"
//#include "MfdSurfaceReconstruction/CIsoSurface.h"
//#include "MfdMath/LinearSystem/MeshlessSolver.h"

//#include "Eigen/Sparse"
//#include "Eigen/IterativeLinearSolvers"
//#include "Eigen/Core"
#include<stdlib.h>
#include<string.h>
#include"vector_types.h"






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

#define RHO1 1000.0f
#define RHO2 10.0f

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
		template<typename T>
		void UpdatePhi(Grid<3, T>& d, Grid<3, T>& d0, float dt);
		void SetScalarFieldBoundary(Grid3f& field, bool postive);
		void LinearSolve(Grid3f& d, Grid3f& d0, float a, float c);
		void MarkSolidDomain();
		void AllocateMemoery(int _np, int _nx, int _ny, int _nz);

		template<typename T>
		void AdvectForward(Grid<3, T>& d, Grid<3, T>& d0, GridV3f& v, float dt);
		void Invoke(unsigned char type, unsigned char key, int x, int y);
		//virtual void PostProcessing();

		float CFL();



		/*template<typename T>
		void AdvectForward(Grid<3, T>& d, Grid<3, T>& d0, GridV3f& v, float dt);*/
		void cudaInitialSeparation();
		void cudaMarkSolidDomain();
		void cudavelGrid_phase(GridV3f velGrid_phase, float t);
		void cudaAllocateMemoery();
		void cudamassGrid_phase(Grid3f& massGrid_phase, GridV3f& posGrid_Air, Vector3f origin, int row, int col, int depth);
		void cudaAllocateMemoery(int Nx, int Ny, int Nz);
		//virtual void StepEuler(float dt);
		void cudaAdvectForward(Grid3f d, Grid3f d0, GridV3f v, float dt);
		void cudaUpdatePhi(Grid3f& device_d, Grid3f& device_d0, GridV3f v, float dt);
		void cudaLinerSolve(Grid3f& phi, Grid3f& phi0, float c);
		void cudaSetScalarFieldBoundary(Grid3f& Device_field, bool postive);
		void cudaSetScalarFieldBoundary1(float* field, bool postive);
		float cudaCFL();



		//2019/8/27
		//N-S equation
		void SetU(Grid3f vel_u);
		void SetV(Grid3f vel_v);
		void SetW(Grid3f vel_w);
		void InitVolecity(Grid3f mass, Grid3f vel_u1, Grid3f vel_v1, Grid3f vel_w1, float dt);
		void InterpolateVelocity(Grid3f vel_u, Grid3f vel_v, Grid3f vel_w, float dt);
		void ApplyGravityForce(Grid3f vel_u, Grid3f vel_v, Grid3f vel_w, Grid3f mass, float dt);
		void PrepareForProjection(GridCoef coefMatrix, Grid3f RHS, Grid3f vel_u, Grid3f vel_v, Grid3f vel_w,Grid3f mass, float dt);
		void Projection(Grid3f pressure, GridCoef coefMatrix, Grid3f RHS, int numIter, float dt);
		void UpdateVelocity(Grid3f vel_u, Grid3f vel_v, Grid3f vel_w,Grid3f press, Grid3f mass_host, float dt);
		void AdvectionVelocity(GridV3f vel, Grid3f vel_u, Grid3f vel_v, Grid3f vel_w, float dt);


		//2019/10/18
		//Multigrid

		void Restriction(Grid3f src, Grid3f std);
		void Interplation(Grid3f src, Grid3f std);
		void Jacobi(Grid3f pressure, Grid3f buf, GridCoef coefMatrix, Grid3f RHS);
		void Correction(Grid3f r1, Grid3f b1, Grid3f A1, Grid3f u1);
		void Multigrid(Grid3f A, Grid3f b, Grid3f V, int k);
		void CorrectionOnCarse(Grid3f u, Grid3f v);


	public:
		
		//ApplyGravity
		Grid3f D_Gravity_velu;
		Grid3f D_Gravity_velv;
		Grid3f D_Gravity_velw;

		//Advection
		Grid3f D_Advection_velu;
		Grid3f D_Advection_velv;
		Grid3f D_Advection_velw;
		GridV3f D_Advection_uvw;
		GridV3f D_Advection_uvw1;

		//Divergence
		Grid3f D_Divergence_velu;
		Grid3f D_Divergence_velv;
		Grid3f D_Divergence_velw;
		Grid3f D_Divergence_mass;
		Grid3f D_Divergence_RHS;
		GridCoef D_Divergence_coefMatrix;
		
		//Projection
		Grid3f temp;
		Grid3f D_Projection_RHS;
		Grid3f D_Projection_buf;
		Grid3f D_Projection_pressure;
		GridCoef D_Projection_coefMatrix;

		//Updata velecity
		Grid3f D_Updata_velu;
		Grid3f D_Updata_velv;
		Grid3f D_Updata_velw;
		Grid3f D_Updatamass;
		Grid3f D_Updatapressure;


		Grid3f H_buf;
		Grid3f Devmass1;
		Grid3f Host_pressure;
		GridCoef coefMatrix;
		Grid3f RHS;
		int numIter;

		Grid3f velHost_u;
		Grid3f velHost_v;
		Grid3f velHost_w;

		Grid3f Host_dataP;



		//PrepareForProjection





	

		Grid3f vel_u_boundary;
		Grid3f vel_v_boundary;
		Grid3f vel_w_boundary;

		Grid3f pre_vel_u;
		Grid3f pre_vel_v;
		Grid3f pre_vel_w;
		Grid<3, Coeff> coef_u;
		Grid<3, Coeff> coef_v;
		Grid<3, Coeff> coef_w;


		//Eigen::VectorXd x0;
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
		int nx;
		int ny;
		int nz;
		int dsize;
		//int Nx;
		//int Ny;
		//int Nz;

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

	
		float* device_ren_massfield;
		Grid3f device_preMassGrid_Air;
		Grid3f device_massGrid_phase;
		float3* dnGrid;

		Grid3f Ddevice_d;
		Grid3f Ddevice_d0;
		GridV3f Device_v;

		Grid3f device_vel_u;
		Grid3f device_vel_v;
		GridV3f device_velGrid_phase;
		//float3* device_posGrid_Air;
		//float* device_massGrid_phase;
		//float* preMassGrid_Air;
		float samplingDistance;
		float* device_vel_u_boundary;
		float* device_vel_v_boundary;
		float* device_vel_w_boundary;
		GridV3f device_posGrid_Air;
		Grid3f Device_field;

		//Grid3f device_d0;
		//Grid3f device_d;

		Grid3f Device_L_phi;
		Grid3f Device_L_phi0;
		Grid3f  Device_cp;



	};
}