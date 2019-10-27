#include"cuda_runtime.h"
#include <iostream>
//#include <helper_cuda.h>
//#include <helper_functions.h> // helper utility functions 
#include"device_functions.h"
#include "cuda.h"
#include "HybridMultiPhaseFluid.h"
#include<device_launch_parameters.h>
#include <sstream>
#include<algorithm>
#include<cmath>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <sstream>
#include "MfdIO/Image.h"
#include "MfdNumericalMethod/UniformGridQuery.h"
#include<fstream>
#include "pcgsolver/pcg_solver.h"



namespace mfd {
	using namespace std;
#define FOR_EACH_CELL for (int i=1 ; i<Nx-1 ; i++ ) { for (int j=1 ; j<Ny-1 ; j++ ) { for (int k=1 ; k<Nz-1 ; k++ ) {
#define END_FOR }}}
	//using namespace std;
	//#define dsize Nx*Ny*Nz
	//#define Nx 512
	//#define Ny 512
#define BLOCK_SIZE 16
//#define EPSILON 1e-6
	__constant__ PFParameter pfParams;
#ifdef NDEBUG
#define cuSynchronize() {}
#else
#define cuSynchronize()	{						\
		char str[200];							\
		cudaDeviceSynchronize();				\
		cudaError_t err = cudaGetLastError();	\
		if (err != cudaSuccess)					\
		{										\
			sprintf(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);		\
			throw std::runtime_error(std::string(str));																\
		}																											\
	}
#endif
#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
	//int nx;
	//int ny;
	//int nz;
	//int dsize;

	float diff = 0.0f;
	float h = 0.005f;
	float gamma = 1.0f;
	float ceo1 = 1.0f*gamma / h;		//for smoothing


	__global__ void K_CopyData(Grid3f dst, Grid3f src)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= src.nx) return;
		if (j >= src.ny) return;
		if (k >= src.nz) return;
		//if (i >= 1 && i < dst.nx - 1 && j >= 1 && j < dst.ny - 1 && k >= 1 && k < dst.nz - 1)
		//{
			dst(i, j, k) = src(i, j, k);
		//}
	}



//Phase field equation

/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/
/*-------------------------------Phase field equation------------------------- */
/*-----------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------*/
/*------------------------------------Initialize ----------------------------- */
/*-----------------------------------------------------------------------------*/
	HybridMultiPhaseFluid::HybridMultiPhaseFluid(void)
	{
		Initialize();
	}

	bool HybridMultiPhaseFluid::Initialize(string in_filename /*= "NULL" */)
	{
		Config::reload("config3d.txt");
		cout << "Total Liquid Particle: " << 15 * 15 * 15 << endl;

		timeStep = Config::timestep;//时间步
		samplingDistance = Config::samplingdistance;//采样距离
		//smoothingLength = Config::samplingdistance*Config::smoothinglength;//光滑半径/长度
		//viscosityAir = Config::viscosity;//空气粘度
		rhoLiquidRef = Config::density;//流体密度
		//V_grid = pow(samplingDistance, 3);
		rhoAirRef = rhoLiquidRef / 4; //空气密度

		massAir = rhoAirRef * samplingDistance*samplingDistance*samplingDistance;//空气质量
		massLiquid = rhoLiquidRef * samplingDistance*samplingDistance*samplingDistance;//流体质量
		cout << "Liquid Particle Mass: " << massLiquid << endl;
		cout << "Liquid Density: " << rhoLiquidRef << endl;
		cout << "Air Particle Mass: " << massAir << endl;
		cout << "Air Density: " << rhoAirRef << endl;
		cout << "smoothinglength" << Config::smoothinglength << endl;
		diff = Config::diffusion;

#ifdef DEMO_SEPERATION
		InitialSeparation();
		//cudaInitialSeparation();
#endif
		for (int i = 1; i < nx - 1; i++)
		{
			for (int j = 1; j < ny - 1; j++)
			{
				for (int k = 1; k < nz - 1; k++)
				{
					marker_phase[0](i, j, k) = true;
				}
			}
		}
		MarkSolidDomain();
		ren_massfield = (massGrid_phase[0].data);
		ren_mass = 1.0f;
		ren_marker = (marker_phase[0].data);
		return true;
	}


	void HybridMultiPhaseFluid::InitialSeparation()
	{
		int valid_x = Config::dimX;
		int valid_y = Config::dimY;
		int valid_z = Config::dimZ;

		int half_row = valid_x + 1;
		int half_col = valid_y + 1;
		int half_depth = valid_z + 1;

		int row = 2 * half_row + 1;
		int col = 2 * half_col + 1;
		int depth = 2 * half_depth + 1;
		int liquid_num = 0;

		origin = Vector3f(0.5f - half_row * samplingDistance, 0.5f - half_col * samplingDistance, 0.5f - half_depth * samplingDistance);
		AllocateMemoery(liquid_num, row, col, depth);

		int pid = 0;
		int num_air = 0;

		cudamassGrid_phase(massGrid_phase[0], posGrid_Air, origin, row, col, depth);
		Vector3f end = Vector3f(0.5f + half_row * samplingDistance, 0.5f + half_col * samplingDistance, 0.5f + half_depth * samplingDistance);
	}


	__global__ void Kernel_InitialSeparation(GridV3f posGrid_Air, Vector3f origin, int nx, int ny, int nz)
	{
		int index;
		float samplingDistance = 0.005f;
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (i >= 0 && i < nx  && j >= 0 && j < ny && k >= 0 && k < nz)
		{
			float x = origin.x + i * samplingDistance;
			float y = origin.y + j * samplingDistance;
			float z = origin.z + k * samplingDistance;
			if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
			{
				posGrid_Air(i, j, k) = Vector3f(x, y, z);

			}
		}

	}

	__global__ void Kernel_massGrid_phase(Grid3f massGrid_phase1, GridV3f posGrid_Air, int nx, int ny, int nz)
	{
		int index;
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			index = i + j*nx + k*nx*ny;

			Vector3f center = Vector3f(0.0f, 0.0f, -0.5f);//这个是球的坐标
														  //float d = (posGrid_Air(i, j, k) - center).Length();					  //df->GetDistance(posGrid_Air(i, j, k), d);
														  //float d = sqrt(pow(posGrid_Air[index].x-center.x,2)+pow(posGrid_Air[index].y-center.y,2)+pow(posGrid_Air[index].z-center.z,2));//点到中心的距离
														  //if (d < 0.15f)// && !(posGrid_Air(i,j,k).x < 0.53f && posGrid_Air(i,j,k).x > 0.47f && posGrid_Air(i,j,k).y < 0.83f))
			if (i > 5 && i < 35 && j > 5 && j < 75)
			{
				massGrid_phase1(i, j, k) = 1.0f;
			}
			else
			{
				massGrid_phase1(i, j, k) = 0.0f;
			}
		}
	}

	void HybridMultiPhaseFluid::cudamassGrid_phase(Grid3f& massGrid_phase, GridV3f& posGrid_Air, Vector3f origin, int row, int col, int depth)
	{

		device_posGrid_Air.cudaSetSpace(nx, ny, nz);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);

		Kernel_InitialSeparation << <dimGrid, dimBlock >> > (device_posGrid_Air, origin, row, col, depth);
		cuSynchronize();
		device_posGrid_Air.CopyFromDeviceToHost(posGrid_Air);

		Kernel_massGrid_phase << <dimGrid, dimBlock >> > (device_massGrid_phase, device_posGrid_Air, row, col, depth);
		cuSynchronize();
		device_massGrid_phase.CopyFromDeviceToHost(massGrid_phase);
	}





	//Using
	void HybridMultiPhaseFluid::MarkSolidDomain()
	{
		marker_Solid.Zero();
		int gNum = GetAirParticleNumber();
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				for (int k = 0; k < nz; k++)
				{
					if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
					{
						marker_Solid(i, j, k) = true;
					}
					else
						marker_Solid(i, j, k) = false;
				}
			}
		}
	}


	//Using--内存分配
	void HybridMultiPhaseFluid::AllocateMemoery(int _np, int _nx, int _ny, int _nz)
	{
		BasicSPH::AllocMemory(_np, 1, 0);
		liquidNeigbhors.SetSpace(_np);
		airNeigbhors.SetSpace(_np);
		rhoAirArr.SetSpace(_np);
		phiLiquid.SetSpace(_np);
		energyLiquid.SetSpace(_np);

		velHost_u.SetSpace(_nx + 1, _ny, _nz);
		velHost_v.SetSpace(_nx, _ny + 1, _nz);
		velHost_w.SetSpace(_nx, _ny, _nz + 1);
		coefMatrix.SetSpace(_nx , _ny, _nz);
		RHS.SetSpace(_nx, _ny, _nz);
		Host_pressure.SetSpace(_nx, _ny, _nz);
		Host_dataP.SetSpace(_nx, _ny, _nz);

		vel_u_boundary.SetSpace(_nx + 1, _ny, _nz);
		vel_v_boundary.SetSpace(_nx, _ny + 1, _nz);
		vel_w_boundary.SetSpace(_nx, _ny, _nz + 1);

		H_buf.SetSpace(_nx, _ny, _nz);

		pre_vel_u.SetSpace(_nx + 1, _ny, _nz);
		pre_vel_v.SetSpace(_nx, _ny + 1, _nz);
		pre_vel_w.SetSpace(_nx, _ny, _nz + 1);
		coef_u.SetSpace(_nx + 1, _ny, _nz);
		coef_v.SetSpace(_nx, _ny + 1, _nz);
		coef_w.SetSpace(_nx, _ny, _nz + 1);
		marker_Air.SetSpace(_nx, _ny, _nz);
		marker_Solid.SetSpace(_nx, _ny, _nz);

		posGrid_Air.SetSpace(_nx, _ny, _nz);
		preMassGrid_Air.SetSpace(_nx, _ny, _nz);

		p.SetSpace(_nx, _ny, _nz);
		divu.SetSpace(_nx, _ny, _nz);
		for (int i = 0; i < PHASE_SIZE; i++)
		{
			massGrid_phase[i].SetSpace(_nx, _ny, _nz);
			velGrid_phase[i].SetSpace(_nx, _ny, _nz);
			marker_phase[i].SetSpace(_nx, _ny, _nz);

		}

		nx = _nx;
		ny = _ny;
		nz = _nz;
		dsize = nx*ny*nz;
		cudaAllocateMemoery(nx,ny,nz);//Allocation memory on CUDA

	}

	template<class T>
	T sqr(const T& x)
	{
		return x*x;
	}

	void HybridMultiPhaseFluid::ProjectConstantMAC(Grid3f vel_u, Grid3f vel_v, Grid3f vel_w, Grid3f mass, float h, float rho1, float rho2, float dt)
	{
		PCGSolver<double> solver;
		SparseMatrixd matrix;
		std::vector<double> rhs;
		std::vector<double> pressure;

		int ni = nx;
		int nj = ny;
		int nk = nz;


		int system_size = ni*nj*nk;

		rhs.resize(system_size);
		pressure.resize(system_size);
		matrix.resize(system_size);

		matrix.zero();
		rhs.assign(rhs.size(), 0);
		pressure.assign(pressure.size(), 0);

		//Build the linear system for pressure

		for (int k = 1; k < nk - 1; ++k) {
			for (int j = 1; j < nj - 1; ++j) {
				for (int i = 1; i < ni - 1; ++i) {
					int index = i + ni*j + ni*nj*k;

					rhs[index] = 0;
					pressure[index] = 0;
					//right neighbour

					if (i < nx - 2) {
						float c = 0.5f*(mass(i, j, k) + mass(i + 1, j, k));
						c = c > 1.0f ? 1.0f : c;
						c = c < 0.0f ? 0.0f : c;
						float term = dt / sqr(h) / (rho1*c + rho2*(1.0f - c));

						matrix.add_to_element(index, index, term);
						matrix.add_to_element(index, index + 1, -term);
					}
					rhs[index] -= vel_u(i + 1, j, k) / h;

					//left neighbour

					if (i > 1) {
						float c = 0.5f*(mass(i, j, k) + mass(i - 1, j, k));
						c = c > 1.0f ? 1.0f : c;
						c = c < 0.0f ? 0.0f : c;
						float term = dt / sqr(h) / (rho1*c + rho2*(1.0f - c));
						matrix.add_to_element(index, index, term);
						matrix.add_to_element(index, index - 1, -term);
					}
					rhs[index] += vel_u(i, j, k) / h;

					//top neighbour

					if (j < nj - 2) {
						float c = 0.5f*(mass(i, j, k) + mass(i, j + 1, k));
						c = c > 1.0f ? 1.0f : c;
						c = c < 0.0f ? 0.0f : c;
						float term = dt / sqr(h) / (rho1*c + rho2*(1.0f - c));
						matrix.add_to_element(index, index, term);
						matrix.add_to_element(index, index + ni, -term);
					}
					rhs[index] -= vel_v(i, j + 1, k) / h;

					//bottom neighbour

					if (j > 1) {
						float c = 0.5f*(mass(i, j, k) + mass(i, j - 1, k));
						c = c > 1.0f ? 1.0f : c;
						c = c < 0.0f ? 0.0f : c;
						float term = dt / sqr(h) / (rho1*c + rho2*(1.0f - c));
						matrix.add_to_element(index, index, term);
						matrix.add_to_element(index, index - ni, -term);
					}
					rhs[index] += vel_v(i, j, k) / h;


					//far neighbour

					if (k < nz - 2) {
						float c = 0.5f*(mass(i, j, k) + mass(i, j, k + 1));
						c = c > 1.0f ? 1.0f : c;
						c = c < 0.0f ? 0.0f : c;
						float term = dt / sqr(h) / (rho1*c + rho2*(1.0f - c));
						matrix.add_to_element(index, index, term);
						matrix.add_to_element(index, index + ni*nj, -term);
					}
					rhs[index] -= vel_w(i, j, k + 1) / h;

					//near neighbour

					if (k > 1) {
						float c = 0.5f*(mass(i, j, k) + mass(i, j, k - 1));
						c = c > 1.0f ? 1.0f : c;
						c = c < 0.0f ? 0.0f : c;
						float term = dt / sqr(h) / (rho1*c + rho2*(1.0f - c));
						matrix.add_to_element(index, index, term);
						matrix.add_to_element(index, index - ni*nj, -term);
					}
					rhs[index] += vel_w(i, j, k) / h;
				}
			}
		}

		double tolerance;
		int iterations;
		solver.set_solver_parameters(1e-6, 500);
		bool success = solver.solve(matrix, rhs, pressure, tolerance, iterations);
		//printf("Solver took %d iterations and had residual %e\n", iterations, tolerance);
		if (!success) {
			printf("WARNING: Pressure solve failed!************************************************\n");
		}
		else
			cout << "Succeded! Interation number: " << iterations << " Threshold: " << tolerance << endl;

		float nij = ni*nj;

#pragma omp parallel for
		for (int i = 2; i < vel_u.nx - 2; i++)
		{
			for (int j = 1; j < vel_u.ny - 1; j++)
			{
				for (int k = 1; k < vel_u.nz - 1; k++)
				{
					int index = i + j*ni + k*ni*nj;
					float c = 0.5f*(mass[index - 1] + mass[index]);
					c = c > 1.0f ? 1.0f : c;
					c = c < 0.0f ? 0.0f : c;

					vel_u(i, j, k) -= dt*(pressure[index] - pressure[index - 1]) / h / (c*rho1 + (1.0f - c)*rho2);
				}
			}
		}

#pragma omp parallel for
		for (int i = 1; i < vel_v.nx - 1; i++)
		{
			for (int j = 2; j < vel_v.ny - 2; j++)
			{
				for (int k = 1; k < vel_v.nz - 1; k++)
				{
					int index = i + j*ni + k*ni*nj;
					float c = 0.5f*(mass[index] + mass[index - ni]);
					c = c > 1.0f ? 1.0f : c;
					c = c < 0.0f ? 0.0f : c;

					vel_v(i, j, k) -= dt*(pressure[index] - pressure[index - ni]) / h / (c*rho1 + (1.0f - c)*rho2);
				}
			}
		}

#pragma omp parallel for
		for (int i = 1; i < vel_w.nx - 1; i++)
		{
			for (int j = 1; j < vel_w.ny - 1; j++)
			{
				for (int k = 2; k < vel_w.nz - 2; k++)
				{
					int index = i + j*ni + k*ni*nj;
					float c = 0.5f*(mass[index] + mass[index - nij]);
					c = c > 1.0f ? 1.0f : c;
					c = c < 0.0f ? 0.0f : c;
					vel_w(i, j, k) -= dt*(pressure[index] - pressure[index - nij]) / h / (c*rho1 + (1.0f - c)*rho2);
				}
			}
		}
	}


	/*-----------------------------------------------------------------------------*/
	/*------------------------------------Start Euler ---------------------------- */
	/*-----------------------------------------------------------------------------*/
	static float t = -0.01f;
	//Using--
	void HybridMultiPhaseFluid::StepEuler(float dt)
	{
		clock_t total_start = clock();
		clock_t t_start = clock();
		clock_t t_end;

		float elipse = 0.0f;
		float dx = 1.0f / nx;
		float T = 4.0f;
		
		t_end = clock();
		cout << "Solving Pressure Costs: " << t_end - t_start << endl;
		t_start = clock();
		while (elipse < dt) {
			float substep = CFL();
			if (elipse + substep > dt)
			{
				substep = dt - elipse;
			}
			cout << "*********Substep: " << substep << " *********" << endl;
			t_start = clock();
			//----------------------solving N-S equation -------------------------//
			//InitVolecity(massGrid_phase[0], vel_hu, vel_hv, vel_hw, dt);
			ApplyGravityForce(velHost_u, velHost_v, velHost_w, massGrid_phase[0], substep);
			InterpolateVelocity(velHost_u, velHost_v, velHost_w, substep);

			for (size_t i = 0; i < nx; i++)
			{
				for (size_t j = 0; j < ny; j++)
				{
					velHost_w(i, j, 0) = 0.0f;
					velHost_w(i, j, 1) = 0.0f;
					velHost_w(i, j, nz) = 0.0f;
					velHost_w(i, j, nz - 1) = 0.0f;
				}
			}

			for (size_t j = 0; j < ny; j++)
			{
				for (size_t k = 0; k < nz; k++)
				{
					velHost_u(0, j, k) = 0.0f;
					velHost_u(1, j, k) = 0.0f;
					velHost_u(nx, j, k) = 0.0f;
					velHost_u(nx - 1, j, k) = 0.0f;
				}
			}

			for (size_t k = 0; k < nz; k++)
			{
				for (size_t i = 0; i < nx; i++)
				{
					velHost_v(i, 0, k) = 0.0f;
					velHost_v(i, 1, k) = 0.0f;
					velHost_v(i, ny, k) = 0.0f;
					velHost_v(i, ny - 1, k) = 0.0f;
				}
			}

			//for (size_t i = 0; i < massGrid_phase[0].elementCount; i++)
			//{
			//	massGrid_phase[0][i] = 1.0f;
			//}

			//velHost_u.Zero();
			//velHost_v.Zero();
			//velHost_w.Zero();
			ProjectConstantMAC(velHost_u, velHost_v, velHost_w, massGrid_phase[0], h, RHO2, RHO1, substep);

			//PrepareForProjection(coefMatrix, RHS, velHost_u, velHost_v, velHost_w, massGrid_phase[0], substep);
			//Projection(Host_pressure, Host_dataP,H_buf,coefMatrix, RHS, 300, substep);
			//UpdateVelocity(velHost_u, velHost_v, velHost_w, Host_pressure, massGrid_phase[0], substep);
			
			//float pd;
			//ofstream  fout("vel_gv.txt");   //创建一个文件
			//for (int i = 0; i < nx; i++)
			//{
			//	for (int j = 0; j < ny; j++)
			//	{
			//		for (int k = 0; k < nz; k++)
			//		{
			//			fout << "(" << i << "," << j << "," << k << ",)" << "=" << RHS(i, j, k) << endl;
			//		}
			//	}
			//}
			for (int i = 1; i < nx - 1; i++)
			{
				for (int j = 1; j < ny - 1; j++)
				{
					for (int k = 1; k < nz - 1; k++)
					{
						velGrid_phase[0](i, j, k)[0] = 0.5f*(velHost_u(i - 1, j, k) + velHost_u(i, j, k));
						velGrid_phase[0](i, j, k)[1] = 0.5f*(velHost_v(i, j - 1, k) + velHost_v(i, j, k));
						velGrid_phase[0](i, j, k)[2] = 0.5f*(velHost_w(i, j, k - 1) + velHost_w(i, j, k));
					}
				}
			}
			
			preMassGrid_Air = massGrid_phase[0];
			AdvectWENO1rd(massGrid_phase[0], preMassGrid_Air, velHost_u, velHost_v, velHost_w, substep);
			//cudaAdvectForward(massGrid_phase[0], preMassGrid_Air, velGrid_phase[0], substep);
			t_end = clock();
			cout << "Advect Time: " << t_end - t_start << endl;
			t_start = clock();
			t_end = clock();
			cout << "Update Time: " << t_end - t_start << endl;
			elipse += substep;
		}

		//float totalMass = 0.0f;
		//for (int i = 0; i < massGrid_phase[0].elementCount; i++)
		//{
		//	totalMass += massGrid_phase[0][i];
		//}
		//cout << "**********Total Mass: " << totalMass / massGrid_phase[0].elementCount;


		cout << dt << endl;
		t += dt;
		clock_t total_end = clock();
		cout << "Total Cost " << total_end - total_start << " million seconds!" << endl;

		if (simItor*dt > 4.01f)
		{
			exit(0);
		}

	}




	/*-----------------------------------------------------------------------------*/
	/*-----------------------------------CFL condition---------------------------- */
	/*-----------------------------------------------------------------------------*/
#define INNERINDEX(m,n,l) (m-1)*(Ny-2)*(Nz-2)+(n-1)*(Nz-2)+l-1
//Using-- Cfl condition
	float HybridMultiPhaseFluid::CFL()
	{
		float maxvel = 0.0f;
		for (int i = 0; i < velHost_u.Size(); i++)
			maxvel = max(maxvel, abs(velHost_u[i]));
		for (int i = 0; i < velHost_v.Size(); i++)
			maxvel = max(maxvel, abs(velHost_v[i]));
		for (int i = 0; i < velHost_w.Size(); i++)
			maxvel = max(maxvel, abs(velHost_w[i]));
		if (maxvel < EPSILON)
			maxvel = 1.0f;
		return samplingDistance / maxvel;
	}



	/*-----------------------------------------------------------------------------*/
	/*-----------------------------------AdvectForward---------------------------- */
	/*-----------------------------------------------------------------------------*/
	__global__ void Kenel_AdvectForward(Grid3f phi, Grid3f phi0, GridV3f v, int nx, int ny, int nz, float dt)
	{
		float h = 0.005f;
		float fx, fy, fz;
		int  ix, iy, iz;
		float w000, w100, w010, w001, w111, w011, w101, w110;
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			int idx = i + j*nx + k*nx*ny;
			fx = i + dt*v(i, j, k).x / h;
			fy = j + dt*v(i, j, k).y / h;
			fz = k + dt*v(i, j, k).z / h;
			if (fx < 1) { fx = 1; }
			if (fx > nx - 2) { fx = nx - 2; }
			if (fy < 1) { fy = 1; }
			if (fy > ny - 2) { fy = ny - 2; }
			if (fz < 1) { fz = 1; }
			if (fz > nz - 2) { fz = nz - 2; }

			ix = (int)fx;
			iy = (int)fy;
			iz = (int)fz;
			fx -= ix;
			fy -= iy;
			fz -= iz;
			float& val = phi0[idx];
			//float& val = phi0(i,j,k);
			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			w100 = fx * (1.0f - fy)*(1.0f - fz);
			w010 = (1.0f - fx)*fy *(1.0f - fz);
			w001 = (1.0f - fx)*(1.0f - fy)*fz;
			w111 = fx * fy * fz;
			w011 = (1.0f - fx)*fy * fz;
			w101 = fx * (1.0f - fy)*fz;
			w110 = fx * fy *(1.0f - fz);
			//原子操作
			atomicAdd(&phi(ix, iy, iz), val * w000);
			atomicAdd(&phi(ix + 1, iy, iz), val * w100);
			atomicAdd(&phi(ix, iy + 1, iz), val * w010);
			atomicAdd(&phi(ix, iy, iz + 1), val * w001);

			atomicAdd(&phi(ix + 1, iy + 1, iz + 1), val * w111);
			atomicAdd(&phi(ix, iy + 1, iz + 1), val * w011);
			atomicAdd(&phi(ix + 1, iy, iz + 1), val * w101);
			atomicAdd(&phi(ix + 1, iy + 1, iz), val * w110);

		}
	}

	void HybridMultiPhaseFluid::SetScalarFieldBoundary1(Grid3f& field, bool postive)
	{
		int nx = field.Nx();
		int ny = field.Ny();
		int nz = field.Nz();

		float s = postive ? 1.0f : -1.0f;

#pragma omp parallel for
		for (int j = 1; j < ny - 1; j++)
		{
			for (int k = 1; k < nz - 1; k++)
			{
				field(0, j, k) = s*field(1, j, k);
				field(nx - 1, j, k) = s*field(nx - 2, j, k);
			}
		}

#pragma omp parallel for
		for (int i = 1; i < nx - 1; i++)
		{
			for (int k = 1; k < nz - 1; k++)
			{
				field(i, 0, k) = s*field(i, 1, k);
				field(i, ny - 1, k) = s*field(i, ny - 2, k);
			}
		}

#pragma omp parallel for
		for (int i = 1; i < nx - 1; i++)
		{
			for (int j = 1; j < ny - 1; j++)
			{
				field(i, j, 0) = s*field(i, j, 1);
				field(i, j, nz - 1) = s*field(i, j, nz - 2);
			}
		}

		for (int i = 1; i < nx - 1; i++)
		{
			field(i, 0, 0) = 0.5f*(field(i, 1, 0) + field(i, 0, 1));
			field(i, ny - 1, 0) = 0.5f*(field(i, ny - 2, 0) + field(i, ny - 1, 1));
			field(i, 0, nz - 1) = 0.5f*(field(i, 1, nz - 1) + field(i, 0, nz - 2));
			field(i, ny - 1, nz - 1) = 0.5f*(field(i, ny - 1, nz - 2) + field(i, ny - 2, nz - 1));
		}

		for (int j = 1; j < ny - 1; j++)
		{
			field(0, j, 0) = 0.5f*(field(1, j, 0) + field(0, j, 1));
			field(0, j, nz - 1) = 0.5f*(field(1, j, nz - 1) + field(0, j, nz - 2));
			field(nx - 1, j, 0) = 0.5f*(field(nx - 2, j, 0) + field(nx - 2, j, 1));
			field(nx - 1, j, nz - 1) = 0.5f*(field(nx - 2, j, nz - 1) + field(nx - 1, j, nz - 2));
		}

		for (int k = 1; k < nz - 1; k++)
		{
			field(0, 0, k) = 0.5f*(field(1, 0, k) + field(0, 1, k));
			field(nx - 1, 0, k) = 0.5f*(field(nx - 2, 0, k) + field(nx - 1, 1, k));
			field(0, ny - 1, k) = 0.5f*(field(1, ny - 1, k) + field(0, ny - 2, k));
			field(nx - 1, ny - 1, k) = 0.5f*(field(nx - 2, ny - 1, k) + field(nx - 1, ny - 2, k));
		}

		field(0, 0, 0) = (field(1, 0, 0) + field(0, 1, 0) + field(0, 0, 1)) / 3.0f;
		field(0, 0, nz - 1) = (field(1, 0, nz - 1) + field(0, 1, nz - 1) + field(0, 0, nz - 2)) / 3.0f;
		field(0, ny - 1, 0) = (field(1, ny - 1, 0) + field(0, ny - 2, 0) + field(0, ny - 1, 1)) / 3.0f;
		field(nx - 1, 0, 0) = (field(nx - 2, 0, 0) + field(nx - 1, 1, 0) + field(nx - 1, 0, 1)) / 3.0f;
		field(0, ny - 1, nz - 1) = (field(1, ny - 1, nz - 1) + field(0, ny - 2, nz - 1) + field(0, ny - 1, nz - 2)) / 3.0f;
		field(nx - 1, 0, nz - 1) = (field(nx - 2, 0, nz - 1) + field(nx - 1, 1, nz - 1) + field(nx - 1, 0, nz - 2)) / 3.0f;
		field(nx - 1, ny - 1, 0) = (field(nx - 2, ny - 1, 0) + field(nx - 1, ny - 2, 0) + field(nx - 1, ny - 1, 1)) / 3.0f;
		field(nx - 1, ny - 1, nz - 1) = (field(nx - 2, ny - 1, nz - 1) + field(nx - 1, ny - 2, nz - 1) + field(nx - 1, ny - 1, nz - 2)) / 3.0f;
	}

	void HybridMultiPhaseFluid::AdvectWENO1rd(Grid3f& d, Grid3f& d0, Grid3f& u, Grid3f& v, Grid3f& w, float dt)
	{
		d = d0;
		SetScalarFieldBoundary1(d0, true);

		float invh = 1.0f / samplingDistance;

		int nx = d0.Nx();
		int ny = d0.Ny();
		int nz = d0.Nz();


#pragma omp parallel shared(d, d0, u, v, w, dt, invh) num_threads(NUM_THREAD)
		{

#pragma omp for
			for (int i = 1; i < nx - 1; i++)
			{
				for (int j = 1; j < ny - 1; j++)
				{
					for (int k = 1; k < nz - 1; k++)
					{
						float u_mid;
						float c_mid;
						int ix0, iy0, iz0;
						int ix1, iy1, iz1;
						float dc;

						ix0 = i;   iy0 = j; iz0 = k;
						ix1 = i + 1; iy1 = j; iz1 = k;
						if (ix1 < nx - 1)
						{
							u_mid = u(i + 1, j, k);
							if (u_mid > 0.0f)
							{
								c_mid = d0(ix0, iy0, iz0);
							}
							else
							{
								c_mid = d0(ix1, iy1, iz1);
							}
							dc = dt*invh*c_mid*u_mid;
							d(ix0, iy0, iz0) -= dc;
							d(ix1, iy1, iz1) += dc;
						}


						//j and j+1
						ix0 = i; iy0 = j;   iz0 = k;
						ix1 = i; iy1 = j + 1; iz1 = k;
						if (iy1 < ny - 1)
						{
							u_mid = v(i, j + 1, k);
							if (u_mid > 0.0f)
							{
								c_mid = d0(ix0, iy0, iz0);
							}
							else
							{
								c_mid = d0(ix1, iy1, iz1);
							}
							dc = dt*invh*c_mid*u_mid;
							d(ix0, iy0, iz0) -= dc;
							d(ix1, iy1, iz1) += dc;
						}

						ix0 = i; iy0 = j;   iz0 = k;
						ix1 = i; iy1 = j; iz1 = k + 1;
						if (iz1 < nz - 1)
						{
							u_mid = w(i, j, k + 1);
							if (u_mid > 0.0f)
							{
								c_mid = d0(ix0, iy0, iz0);
							}
							else
							{
								c_mid = d0(ix1, iy1, iz1);
							}
							dc = dt*invh*c_mid*u_mid;
							d(ix0, iy0, iz0) -= dc;
							d(ix1, iy1, iz1) += dc;
						}
					}
				}
			}
		}

	}

	void HybridMultiPhaseFluid::cudaAdvectForward(Grid3f d, Grid3f d0, GridV3f v, float dt)
	{
		d.Zero();
		Ddevice_d.cudaSetSpace(nx, ny, nz);
		Ddevice_d0.cudaSetSpace(nx, ny, nz);
		Device_v.cudaSetSpace(nx, ny, nz);
		//copy data
		Ddevice_d0.CopyFromHostToDevice(d0);
		Device_v.CopyFromHostToDevice(v);

		//computer
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		Kenel_AdvectForward << <dimGrid, dimBlock >> > (Ddevice_d, Ddevice_d0, Device_v, nx, ny, nz, dt);
		cuSynchronize();
		Ddevice_d0.CopyFromDeviceToDevice(Ddevice_d);
		cudaUpdatePhi(Ddevice_d, Ddevice_d0, Device_v, dt);
		Ddevice_d.CopyFromDeviceToHost(d);

	}




	/*-----------------------------------------------------------------------------*/
	/*-------------------------------------UpdataPhi------------------------------ */
	/*-----------------------------------------------------------------------------*/

	__global__ void Kenel_LinerSolve(Grid3f phi, Grid3f phi0, Grid3f cp, float c, int Nx, int Ny, int Nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		float c1 = 1.0f / c;
		float c2 = (1.0f - c1) / 6.0f;
		if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
		{
			int k0 = i + j*Nx + k*Nx*Ny;
			phi(i, j, k) = (c1*phi0(i, j, k) + c2*(cp[k0 + 1] + cp[k0 - 1] + cp[k0 + Nx] + cp[k0 - Nx] + cp[k0 + Nx*Ny] + cp[k0 - Nx*Ny]));
		}

	}

	__global__ void Kernel_UpdataPhiNorm(float3* ndGrid, Grid3f dif_field, int Nx, int Ny, int Nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		int index;

		if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
		{
			index = i + j*Nx + k*Nx*Ny;
			float3 norm;
			float norm_x;
			float norm_y;
			float norm_z;
			float eps = 0.000001f;
			norm_x = dif_field[(i + 1) + j*Nx + k*Nx*Ny] - dif_field[(i - 1) + j*Nx + k*Nx*Ny];
			norm_y = dif_field[i + (j + 1)*Nx + k*Nx*Ny] - dif_field[i + (j - 1)*Nx + k*Nx*Ny];
			norm_z = dif_field[i + j*Nx + (k + 1)*Nx*Ny] - dif_field[i + j*Nx + (k - 1)*Nx*Ny];
			float norm_xy = sqrt(norm_x*norm_x + norm_y*norm_y + norm_z*norm_z) + eps;

			//printf("%d:s", dif_field[(i - 1) + j*Nx + k*Nx*Ny]);
			norm_x /= norm_xy;
			norm_y /= norm_xy;
			norm_z /= norm_xy;

			ndGrid[index].x = norm_x;
			ndGrid[index].y = norm_y;
			ndGrid[index].z = norm_z;

		}
	}

	__device__ float K_SharpeningWeight(float dist)
	{
		float fx = dist - floor(dist);
		fx = 1.0f - 2.0f*abs(fx - 0.5f);

		if (fx < 0.01f)
		{
			fx = 0.0f;
		}

		return fx;
	}


	__global__ void Kernel_UpdataPhi(float3* nGrid, Grid3f d0, Grid3f d, GridV3f vel ,float dt, float h, int Nx, int Ny, int Nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		float w = 1.0f*h;
		float gamma = 1.0f;
		float ceo2 = 1.5f*gamma*w / h / h;
		float weight;

		float ceo = 1.5f*gamma / h;
		if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
		{
			int idx = i + j*Nx + k*Nx*Ny;
			int ix0, iy0, iz0;
			int ix1, iy1, iz1;
			int k0, k1;
			ix0 = i;   iy0 = j; iz0 = k;
			ix1 = i + 1; iy1 = j; iz1 = k;
			weight = 1.0f;

			if (ix1 < Nx - 1)
			{
				
				//weight = K_SharpeningWeight(vel_u(i + 1, j, k)*dt / pfParams.h);
				k0 = ix0 + iy0*Nx + iz0*Nx*Ny;
				k1 = ix1 + iy1*Nx + iz1*Nx*Ny;

				//float c0 = d0[k0] * (1.0f - d0[k0])*nGrid[k0].x;
				//float c1 = d0[k1] * (1.0f - d0[k1])*nGrid[k1].x;
				//float dc = 0.5f*weight*ceo2*dt*(c0 + c1);

				//atomicAdd(&d[k0], -dc);
				//atomicAdd(&d[k1], dc);


				float c0;
				float c1;
				float dc;
				c1 = d0[k1] * (1.0f - d0[k1])*nGrid[k1].x;
				c0 = d0[k0] * (1.0f - d0[k0])*nGrid[k0].x;

				dc = 0.5f*(c1 + c0)*ceo*dt;

				//weight = K_SharpeningWeight(vel(i + 1, j, k).x*dt / h);
				atomicAdd(&(d[k0]), -weight*dc);
				atomicAdd(&(d[k1]), weight*dc);



			}

			ix0 = i; iy0 = j; iz0 = k;
			ix1 = i; iy1 = j + 1; iz1 = k;
			if (iy1 < Ny - 1)
			{
				float c0;
				float c1;
				float dc;
				k0 = ix0 + iy0*Nx + iz0*Nx*Ny;
				k1 = ix1 + iy1*Nx + iz1*Nx*Ny;

				//float c0 = d0[k0] * (1.0f - d0[k0])*nGrid[k0].y;
				//float c1 = d0[k1] * (1.0f - d0[k1])*nGrid[k1].y;
				//float dc = 0.5f*weight*ceo2*dt*(c0 + c1);

				//atomicAdd(&d[k0], -dc);
				//atomicAdd(&d[k1], dc);



				c1 = d0[k1] * (1.0f - d0[k1])*nGrid[k1].y;
				c0 = d0[k0] * (1.0f - d0[k0])*nGrid[k0].y;

				dc = 0.5f*(c1 + c0)*ceo*dt;

				//weight = K_SharpeningWeight(vel(i, j + 1, k).y*dt / pfParams.h);
				atomicAdd(&(d[k0]), -weight*dc);
				atomicAdd(&(d[k1]), weight*dc);

			}
			ix0 = i; iy0 = j; iz0 = k;
			ix1 = i; iy1 = j; iz1 = k + 1;
			if (iz1 < Nz - 1)
			{
				float c0;
				float c1;
				float dc;

				k0 = ix0 + iy0*Nx + iz0*Nx*Ny;
				k1 = ix1 + iy1*Nx + iz1*Nx*Ny;

				//float c0 = d0[k0] * (1.0f - d0[k0])*nGrid[k0].z;
				//float c1 = d0[k1] * (1.0f - d0[k1])*nGrid[k1].z;
				//float dc = 0.5f*weight*ceo2*dt*(c0 + c1);
				//atomicAdd(&d[k0], -dc);
				//atomicAdd(&d[k1], dc);


				c1 = d0[k1] * (1.0f - d0[k1])*nGrid[k1].z;
				c0 = d0[k0] * (1.0f - d0[k0])*nGrid[k0].z;

				dc = 0.5f*(c1 + c0)*ceo*dt;

				//weight = K_SharpeningWeight(vel(i, j, k + 1).z*dt / pfParams.h);
				atomicAdd(&(d[k0]), -weight*dc);
				atomicAdd(&(d[k1]), weight*dc);


			}
		}
	}
	void HybridMultiPhaseFluid::cudaUpdatePhi(Grid3f& device_d, Grid3f& device_d0, GridV3f device_v1,float dt)
	{
		cudaSetScalarFieldBoundary(device_d0, true);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		
		Kernel_UpdataPhiNorm << <dimGrid, dimBlock >> > (dnGrid, device_d0, nx, ny, nz);
		cuSynchronize();
		
		Kernel_UpdataPhi << <dimGrid, dimBlock >> > (dnGrid, device_d0, device_d, device_v1,dt, h, nx, ny, nz);
		cuSynchronize();
		//device_d0.CopyFromDeviceToDevice(device_d);
		K_CopyData << <dimGrid, dimBlock >> > (device_d0, device_d);

		float dif2 = (ceo1 + diff / h / h)*dt;
		float c = 1.0f + 6.0f*dif2;
		//Linersolve
		for (int it = 0; it < 20; it++)
		{
			cudaSetScalarFieldBoundary(device_d, true);
			Kenel_LinerSolve << <dimGrid, dimBlock >> > (device_d, device_d0, device_d, c, nx, ny, nz);
			cuSynchronize();
		}
	}




	/*-----------------------------------------------------------------------------*/
	/*--------------------------------SetScalarFieldBoundary---------------------- */
	/*-----------------------------------------------------------------------------*/
	__global__ void SetScalarFieldBoundary_x(Grid3f field, float s, int nx, int ny, int nz)
	{

		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			field(0, j, k) = s * field(1, j, k);
			field(nx - 1, j, k) = s * field(nx - 2, j, k);
		}
	}

	__global__ void SetScalarFieldBoundary_y(Grid3f field, float s, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= 1 && i < nx - 1 && k >= 1 && k < nz - 1)
		{
			field(i, 0, k) = s * field(i, 1, k);
			field(i, ny - 1, k) = s * field(i, ny - 2, k);
		}
	}

	__global__ void SetScalarFieldBoundary_z(Grid3f field, float s, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
		{
			field(i, j, 0) = s * field(i, j, 1);
			field(i, j, nz - 1) = s * field(i, j, nz - 2);
		}
	}

	__global__ void SetScalarFieldBoundary_yz(Grid3f field, float s, int Nx, int Ny, int Nz)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= 0 && i < Nx)
		{
			field(i, 0, 0) = 0.5f*(field(i, 1, 0) + field(i, 0, 1));
			field(i, Ny - 1, 0) = 0.5f*(field(i, Ny - 2, 0) + field(i, Ny - 1, 1));
			field(i, 0, Nz - 1) = 0.5f*(field(i, 1, Nz - 1) + field(i, 0, Nz - 2));
			field(i, Ny - 1, Nz - 1) = 0.5f*(field(i, Ny - 1, Nz - 2) + field(i, Ny - 2, Nz - 1));
		}
	}


	__global__ void SetScalarFieldBoundary_xz(Grid3f field, float s, int Nx, int Ny, int Nz)
	{
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (j >= 1 && j < Ny - 1)
		{
			field(0, j, 0) = 0.5f*(field(1, j, 0) + field(0, j, 1));
			field(0, j, Nz - 1) = 0.5f*(field(1, j, Nz - 1) + field(0, j, Nz - 2));
			field(Nx - 1, j, 0) = 0.5f*(field(Nx - 2, j, 0) + field(Nx - 2, j, 1));
			field(Nx - 1, j, Nz - 1) = 0.5f*(field(Nx - 2, j, Nz - 1) + field(Nx - 1, j, Nz - 2));
		}
	}

	__global__ void SetScalarFieldBoundary_xy(Grid3f field, float s, int Nx, int Ny, int Nz)
	{
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (k >= 1 && k < Nz - 1)
		{
			field(0, 0, k) = 0.5f*(field(1, 0, k) + field(0, 1, k));
			field(Nx - 1, 0, k) = 0.5f*(field(Nx - 2, 0, k) + field(Nx - 1, 1, k));
			field(0, Ny - 1, k) = 0.5f*(field(1, Ny - 1, k) + field(0, Ny - 2, k));
			field(Nx - 1, Ny - 1, k) = 0.5f*(field(Nx - 2, Ny - 1, k) + field(Nx - 1, Ny - 2, k));
		}
	}

	__global__ void Kernel_vel_boundaryx(float* vel_u_boundary, int nx, int ny, int nz)
	{

		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;
		if (j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			vel_u_boundary[0 + j*nx + k*nx*ny] = 0.0f;
			vel_u_boundary[1 + j*nx + k*nx*ny] = 0.0f;
			vel_u_boundary[nx - 1 + j*nx + k*nx*ny] = 0.0f;
			vel_u_boundary[nx - 2 + j*nx + k*nx*ny] = 0.0f;
		}
	}
	__global__ void Kernel_vel_boundaryy(float* vel_v_boundary, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= 1 && i < nx - 1 && k >= 1 && k < nz - 1)
		{
			vel_v_boundary[i + 0 * nx + k*nx*ny] = 0.0f;
			vel_v_boundary[i + 1 * nx + k*nx*ny] = 0.0f;
			if (i < 75 && i > 25)
			{
				vel_v_boundary[i + (ny - 1)*nx + k*nx*ny] = 0.0f;
				vel_v_boundary[i + (ny - 2)*nx + k*nx*ny] = 0.0f;
			}
			else
			{
				vel_v_boundary[i + (ny - 1)*nx + k*nx*ny] = 0.0f;
				vel_v_boundary[i + (ny - 2)*nx + k*nx*ny] = 0.0f;
			}
		}
	}

	__global__ void Kernel_vel_boundaryz(float* vel_w_boundary, int nx, int ny, int nz)
	{

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (i >= 1 && i < ny - 1 && j >= 1 && j < nz - 1)
		{
			vel_w_boundary[i + j*nx + 0 * nx*ny] = 0.0f;
			vel_w_boundary[i + j*nx + 1 * nx*ny] = 0.0f;
			vel_w_boundary[i + j*nx + (nz - 1)*nx*ny] = 0.0f;
			vel_w_boundary[i + j*nx + (nz - 2)*nx*ny] = 0.0f;
		}
	}

	void HybridMultiPhaseFluid::cudaSetScalarFieldBoundary(Grid3f& Device_field, bool postive)
	{
		float s = postive ? 1.0f : -1.0f;
		//computer
		//x=0
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid_x((ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		SetScalarFieldBoundary_x << <dimGrid_x, dimBlock >> > (Device_field, s, nx, ny, nz);
		cuSynchronize();
		//y=0
		dim3 dimGrid_y((nx + dimBlock.x - 1) / dimBlock.x, (nz + dimBlock.z - 1) / dimBlock.z);
		SetScalarFieldBoundary_y << <dimGrid_y, dimBlock >> > (Device_field, s, nx, ny, nz);
		cuSynchronize();
		//z=0
		dim3 dimGrid_z((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y);
		SetScalarFieldBoundary_z << <dimGrid_z, dimBlock >> > (Device_field, s, nx, ny, nz);
		cuSynchronize();

		//xz=0
		dim3 dimGrid_xz((nx + dimBlock.x - 1) / dimBlock.x);
		SetScalarFieldBoundary_xz << <dimGrid_xz, dimBlock >> > (Device_field, s, nx, ny, nz);
		cuSynchronize();
		//yz=0
		dim3 dimGrid_yz((ny + dimBlock.y - 1) / dimBlock.y);
		SetScalarFieldBoundary_yz << <dimGrid_yz, dimBlock >> > (Device_field, s, nx, ny, nz);
		cuSynchronize();
		//xy=0
		dim3 dimGrid_xy((nz + dimBlock.z - 1) / dimBlock.z);
		SetScalarFieldBoundary_xy << <dimGrid_xy, dimBlock >> > (Device_field, s, nx, ny, nz);
		cuSynchronize();
	}












	/*-----------------------------------------------------------------------------*/
	/*-----------------------------------------------------------------------------*/
	/*------------------------------------N-S equation---------------------------- */
	/*-----------------------------------------------------------------------------*/
	/*-----------------------------------------------------------------------------*/
	void HybridMultiPhaseFluid::cudaAllocateMemoery(int nx, int ny, int nz)
	{
		/*mass.cudaSetSpace(nx, ny, nz);*/

		vel_gu.cudaSetSpace(nx + 1, ny, nz);
		vel_gv.cudaSetSpace(nx, ny + 1, nz);
		vel_gw.cudaSetSpace(nx, ny, nz + 1);



		//advection
		vel_du.cudaSetSpace(nx + 1, ny, nz);
		vel_dv.cudaSetSpace(nx, ny + 1, nz);
		vel_dw.cudaSetSpace(nx, ny, nz + 1);
		vel.cudaSetSpace(nx, ny, nz);
		vel_d.cudaSetSpace(nx, ny, nz);


		//solve pressure
		DevcoefMatrix.cudaSetSpace(nx, ny, nz);
		DevRHS.cudaSetSpace(nx, ny, nz);
		Devmass.cudaSetSpace(nx, ny, nz);
		Div_velu.cudaSetSpace(nx + 1, ny, nz);
		Div_velv.cudaSetSpace(nx, ny + 1, nz);
		Div_velw.cudaSetSpace(nx, ny, nz + 1);

		//updataVelecity
		vel_uu.cudaSetSpace(nx + 1, ny, nz);
		vel_uv.cudaSetSpace(nx, ny + 1, nz);
		vel_uw.cudaSetSpace(nx, ny, nz + 1);
		Updatamass.cudaSetSpace(nx, ny, nz);
		Updatapressure.cudaSetSpace(nx, ny, nz);

		//Grid3f buf;
		//buf.SetSpace(nx, ny, nz);
		temp.cudaSetSpace(nx, ny, nz);
		Projection_coefMatrix.cudaSetSpace(nx, ny, nz);
		Projection_RHS.cudaSetSpace(nx, ny, nz);
		Projection_pressure.cudaSetSpace(nx, ny, nz);
		Projection_buf.cudaSetSpace(nx, ny, nz);

		cudaMalloc((void**)&dnGrid, dsize * sizeof(float3));
		device_vel_u.cudaSetSpace(nx + 1, ny, nz);
		device_vel_v.cudaSetSpace(nx, ny + 1, nz);
		device_velGrid_phase.cudaSetSpace(nx, ny, nz);
		device_massGrid_phase.cudaSetSpace(nx, ny, nz);
		cudaMalloc((void**)&device_ren_massfield, dsize * sizeof(float));

	}



	/*-----------------------------------------------------------------------------*/
	/*------------------------------------InitVolecity---------------------------- */
	/*-----------------------------------------------------------------------------*/
	//1、初始化速度场
	__global__ void K_InitVolecity(Grid3f vel_u, Grid3f vel_v, Grid3f vel_w)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = vel_u.nx;
		int ny = vel_v.ny;
		int nz = vel_w.nz;

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		vel_u(i, j, k) = 0.0f;
		vel_v(i, j, k) = 0.0f;
		vel_w(i, j, k) = 0.0f;
	}


	void HybridMultiPhaseFluid::InitVolecity(Grid3f mass1,Grid3f vel_u1, Grid3f vel_v1, Grid3f vel_w1, float dt)
	{
		SetU(vel_u1);
		SetV(vel_v1);
		SetW(vel_w1);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		K_InitVolecity << < dimGrid, dimBlock >> > (vel_u1, vel_v1, vel_w1);

	}

	__global__ void K_SetU(Grid3f vel_u)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = vel_u.nx;
		int ny = vel_u.ny;
		int nz = vel_u.nz;

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		if (i <= 0 || i >= nx - 1)
		{
			vel_u(i, j, k) = 0.0f;
		}
	}
	
	__global__ void K_SetV(Grid3f vel_v)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = vel_v.nx;
		int ny = vel_v.ny;
		int nz = vel_v.nz;

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		if (j <= 0 || j >= ny - 1)
		{
			vel_v(i, j, k) = 0.0f;
		}
	}


	__global__ void K_SetW(Grid3f vel_w)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = vel_w.nx;
		int ny = vel_w.ny;
		int nz = vel_w.nz;

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		if (k <= 0 || k >= nz - 1)
		{
			vel_w(i, j, k) = 0.0f;
		}
	}


	void HybridMultiPhaseFluid::SetU(Grid3f vel_u)
	{
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);

		K_SetU << < dimGrid, dimBlock >> > (vel_u);
	}
	
	void HybridMultiPhaseFluid::SetV(Grid3f vel_v)
	{
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);

		K_SetV << < dimGrid, dimBlock >> > (vel_v);
	}

	void HybridMultiPhaseFluid::SetW(Grid3f vel_w)
	{
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		K_SetW << < dimGrid, dimBlock >> > (vel_w);
	}



	/*-----------------------------------------------------------------------------*/
	/*-------------------------------ApplyGravityForce---------------------------- */
	/*-----------------------------------------------------------------------------*/
	__global__ void K_ApplyGravityForce(Grid3f vel_ku, Grid3f vel_kv, Grid3f vel_kw, int nx, int ny, int nz, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		float g = -9.81f;

		//if (i >= nx) return;
		//if (j >= ny) return;
		//if (k >= nz) return;

		if (i >= 1 && i < nx - 1 && j >= 2 && j < ny - 2 && k >= 1 && k < nz - 1)
		{
		vel_ku(i, j, k) += 0.0f;
		vel_kv(i, j, k) += g*dt;
		vel_kw(i, j, k) += 0.0f;
		//if (j == 0) vel_kv(i, j, k) += -g*dt; return;
		}
	}


	//2
	void HybridMultiPhaseFluid::ApplyGravityForce(Grid3f vel_hu, Grid3f vel_hv, Grid3f vel_hw, Grid3f mass, float dt)
	{

		//ApplyGravity
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);

		K_ApplyGravityForce << < dimGrid, dimBlock >> > (vel_gu, vel_gv, vel_gw,nx, ny, nz, dt);
		cuSynchronize();
		
		vel_gu.CopyFromDeviceToHost(vel_hu);
		vel_gv.CopyFromDeviceToHost(vel_hv);
		vel_gw.CopyFromDeviceToHost(vel_hw);

		//vel_gu.cudaRelease();
		//vel_gv.cudaRelease();
		//vel_gw.cudaRelease();
	}


	/*-----------------------------------------------------------------------------*/
	/*-----------------------------------Advection-------------------------------- */
	/*-----------------------------------------------------------------------------*/
	__global__ void K_InterpolateVelocity(GridV3f vel_k, Grid3f vel_ku, Grid3f vel_kv, Grid3f vel_kw,int nx, int ny, int nz)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		//if (i >= nx) return;
		//if (j >= ny) return;
		//if (k >= nz) return;
		float3 vel_ijk;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			vel_ijk.x = 0.5f*(vel_ku(i, j, k) + vel_ku(i + 1, j, k));
			vel_ijk.y = 0.5f*(vel_kv(i, j, k) + vel_kv(i, j + 1, k));
			vel_ijk.z = 0.5f*(vel_kw(i, j, k) + vel_kw(i, j, k + 1));

			vel_k(i, j, k).x = vel_ijk.x;
			vel_k(i, j, k).y = vel_ijk.y;
			vel_k(i, j, k).z = vel_ijk.z;
		}
	}

	__global__ void K_AdvectionVelocity(GridV3f vel_k, GridV3f vel_k0, int nx, int ny, int nz, float dt)
	{
		float h = 0.005f;
		float fx, fy, fz;
		int  ix, iy, iz;
		float w000, w100, w010, w001, w111, w011, w101, w110;

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		//if (i >= vel_k0.nx) return;
		//if (j >= vel_k0.ny) return;
		//if (k >= vel_k0.nz) return;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{

			fx = i + dt*vel_k0(i, j, k).x / h;
			fy = j + dt*vel_k0(i, j, k).y / h;
			fz = k + dt*vel_k0(i, j, k).z / h;

			if (fx < 1) { fx = 1; }
			if (fx > nx - 2) { fx = nx - 2; }
			if (fy < 1) { fy = 1; }
			if (fy > ny - 2) { fy = ny - 2; }
			if (fz < 1) { fz = 1; }
			if (fz > nz - 2) { fz = nz - 2; }

			ix = (int)fx;
			iy = (int)fy;
			iz = (int)fz;
			fx -= ix;
			fy -= iy;
			fz -= iz;

			//float& val = d0(i,j,k);
			w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
			w100 = fx * (1.0f - fy)*(1.0f - fz);
			w010 = (1.0f - fx)*fy *(1.0f - fz);
			w001 = (1.0f - fx)*(1.0f - fy)*fz;
			w111 = fx * fy * fz;
			w011 = (1.0f - fx)*fy * fz;
			w101 = fx * (1.0f - fy)*fz;
			w110 = fx * fy *(1.0f - fz);
			//原子操作

			//x direction
			atomicAdd(&vel_k(ix, iy, iz).x, vel_k0(i, j, k).x * w000);
			atomicAdd(&vel_k(ix + 1, iy, iz).x, vel_k0(i, j, k).x * w100);
			atomicAdd(&vel_k(ix, iy + 1, iz).x, vel_k0(i, j, k).x * w010);
			atomicAdd(&vel_k(ix, iy, iz + 1).x, vel_k0(i, j, k).x * w001);

			atomicAdd(&vel_k(ix + 1, iy + 1, iz + 1).x, vel_k0(i, j, k).x * w111);
			atomicAdd(&vel_k(ix, iy + 1, iz + 1).x, vel_k0(i, j, k).x * w011);
			atomicAdd(&vel_k(ix + 1, iy, iz + 1).x, vel_k0(i, j, k).x * w101);
			atomicAdd(&vel_k(ix + 1, iy + 1, iz).x, vel_k0(i, j, k).x * w110);

			//y direction
			atomicAdd(&vel_k(ix, iy, iz).y, vel_k0(i, j, k).y * w000);
			atomicAdd(&vel_k(ix + 1, iy, iz).y, vel_k0(i, j, k).y * w100);
			atomicAdd(&vel_k(ix, iy + 1, iz).y, vel_k0(i, j, k).y * w010);
			atomicAdd(&vel_k(ix, iy, iz + 1).y, vel_k0(i, j, k).y * w001);

			atomicAdd(&vel_k(ix + 1, iy + 1, iz + 1).y, -vel_k0(i, j, k).y * w111);
			atomicAdd(&vel_k(ix, iy + 1, iz + 1).y, vel_k0(i, j, k).y * w011);
			atomicAdd(&vel_k(ix + 1, iy, iz + 1).y, vel_k0(i, j, k).y * w101);
			atomicAdd(&vel_k(ix + 1, iy + 1, iz).y, vel_k0(i, j, k).y * w110);

			//z direction
			atomicAdd(&vel_k(ix, iy, iz).z, vel_k0(i, j, k).z * w000);
			atomicAdd(&vel_k(ix + 1, iy, iz).z, vel_k0(i, j, k).z * w100);
			atomicAdd(&vel_k(ix, iy + 1, iz).z, vel_k0(i, j, k).z * w010);
			atomicAdd(&vel_k(ix, iy, iz + 1).z, vel_k0(i, j, k).z * w001);

			atomicAdd(&vel_k(ix + 1, iy + 1, iz + 1).z, vel_k0(i, j, k).z * w111);
			atomicAdd(&vel_k(ix, iy + 1, iz + 1).z, vel_k0(i, j, k).z * w011);
			atomicAdd(&vel_k(ix + 1, iy, iz + 1).z, vel_k0(i, j, k).z * w101);
			atomicAdd(&vel_k(ix + 1, iy + 1, iz).z, vel_k0(i, j, k).z * w110);
		}
	}

	__global__ void K_InterpolatedVelocity(Grid3f vel_ku, Grid3f vel_kv, Grid3f vel_kw, int nx, int ny, int nz, GridV3f vel_k, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		//if (i >= nx) return;
		//if (j >= ny) return;
		//if (k >= nz) return;
		float vx, vy, vz;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			vx = 0.5f*(vel_k(i, j, k).x + vel_k(i + 1, j, k).x);
			vy = 0.5f*(vel_k(i, j, k).y + vel_k(i, j + 1, k).y);
			vz = 0.5f*(vel_k(i, j, k).z + vel_k(i, j, k + 1).z);

			vel_ku(i, j, k) = vx;
			vel_kv(i, j, k) = vy;
			vel_kw(i, j, k) = vz;
		}
	}


	//3、advection
	void HybridMultiPhaseFluid::InterpolateVelocity(Grid3f vel_hu, Grid3f vel_hv, Grid3f vel_hw, float subtep)
	{

		vel_du.CopyFromHostToDevice(vel_hu);
		vel_dv.CopyFromHostToDevice(vel_hv);
		vel_dw.CopyFromHostToDevice(vel_hw);
		

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		
		//Interpolation from Boundary to Center
		K_InterpolateVelocity << < dimGrid, dimBlock >> > (vel, vel_du, vel_dv, vel_dw, nx, ny, nz);
		cuSynchronize();
		//vel_d.Zero();
		//K_CopyData << < dimGrid, dimBlock >> > (vel_d, vel);
		//vel_d.CopyFromDeviceToDevice(vel);
		//Semi-Lagrangian Advection
		K_AdvectionVelocity << < dimGrid, dimBlock >> > (vel_d, vel, nx, ny, nz, subtep);
		cuSynchronize();
		
		//Interpolation from Center to Boundary
		K_InterpolatedVelocity << < dimGrid, dimBlock >> > (vel_du, vel_dv, vel_dw, nx, ny, nz, vel_d, subtep);
		cuSynchronize();

		vel_du.CopyFromDeviceToHost(vel_hu);
		vel_dv.CopyFromDeviceToHost(vel_hv);
		vel_dw.CopyFromDeviceToHost(vel_hw);

		//vel_du.cudaRelease();
		//vel_dv.cudaRelease();
		//vel_dw.cudaRelease();

	}


	/*-----------------------------------------------------------------------------*/
	/*----------------------------------Solve Diffusion---------------------------- */
	/*-----------------------------------------------------------------------------*/
	__global__ void K_PrepareForProjection(GridCoef coefMatrix, Grid3f RHS, Grid3f mass, Grid3f vel_u, Grid3f vel_v, Grid3f vel_w,int nx,int ny,int nz, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		//int nx = mass.nx;
		//int ny = mass.ny;
		//int nz = mass.nz;

		//if (i >= nx) return;
		//if (j >= ny) return;
		//if (k >= nz) return;

		//float h = pfParams.h;
		float h = 0.005f;
		float hh = h*h;

		float div_ijk = 0.0f;
		float S = 0.2f;
		Coef A_ijk;
		
		A_ijk.a = 0.0f;
		A_ijk.x0 = 0.0f;
		A_ijk.x1 = 0.0f;
		A_ijk.y0 = 0.0f;
		A_ijk.y1 = 0.0f;
		A_ijk.z0 = 0.0f;
		A_ijk.z1 = 0.0f;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			float m_ijk = mass(i, j, k);
			if (i + 1 < nx) {
				float c = 0.5f*(m_ijk + mass(i + 1, j, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));//分母是密度，c是phi

				A_ijk.a += term;
				A_ijk.x1 += term;
			}
			div_ijk -= vel_u(i + 1, j, k) / h;
			//left neighbour
			if (i - 1 >= 0) {
				float c = 0.5f*(m_ijk + mass(i - 1, j, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.x0 += term;
			}
			
			div_ijk += vel_u(i, j, k) / h;

			//top neighbour
			if (j + 1 < ny) {
				float c = 0.5f*(m_ijk + mass(i, j + 1, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.y1 += term;
			}

			div_ijk -= vel_v(i, j + 1, k) / h;
			//bottom neighbour
			if (j - 1 >= 0) {
				float c = 0.5f*(m_ijk + mass(i, j - 1, k));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.y0 += term;
			}

			div_ijk += vel_v(i, j, k) / h;
			//far neighbour

			if (k + 1 < nz) {
				float c = 0.5f*(m_ijk + mass(i, j, k + 1));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));
				A_ijk.a += term;
				A_ijk.z1 += term;

			}
			div_ijk -= vel_w(i, j, k + 1) / h;

			//near neighbour
			if (k - 1 >= 0) {
				float c = 0.5f*(m_ijk + mass(i, j, k - 1));
				c = c > 1.0f ? 1.0f : c;
				c = c < 0.0f ? 0.0f : c;
				float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

				A_ijk.a += term;
				A_ijk.z0 += term;
			}
	
			div_ijk += vel_w(i, j, k) / h;
			if (m_ijk > 1.0)
			{
				div_ijk += 0.5f*pow((m_ijk - 1.0f), 1.0f) / dt;
				//div_ijk += S*((mass(i + 1, j, k) - m_ijk)+ (mass(i - 1, j, k) - m_ijk)+ (mass(i, j + 1, k) - m_ijk)+ (mass(i, j - 1, k) - m_ijk) + (mass(i, j, k + 1) - m_ijk)+  (mass(i, j, k - 1) - m_ijk)) / m_ijk / dt;
			}

			coefMatrix(i, j, k) = A_ijk;
			RHS(i, j, k) = div_ijk;//度散
		}
	}
	
	void HybridMultiPhaseFluid::PrepareForProjection(GridCoef coefMatrix, Grid3f RHS,Grid3f vel_hu, Grid3f vel_hv, Grid3f vel_hw,Grid3f mass_host, float subtep)
	{
		Devmass.CopyFromHostToDevice(mass_host);
		Div_velu.CopyFromHostToDevice(vel_hu);
		Div_velv.CopyFromHostToDevice(vel_hv);
		Div_velw.CopyFromHostToDevice(vel_hw);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		K_PrepareForProjection << < dimGrid, dimBlock >> > (DevcoefMatrix, DevRHS, Devmass, Div_velu, Div_velv, Div_velw,nx,ny,nz, subtep);
		cuSynchronize();

		DevcoefMatrix.CopyFromDeviceToHost(coefMatrix);
		DevRHS.CopyFromDeviceToHost(RHS);

		Div_velu.CopyFromDeviceToHost(vel_hu);
		Div_velv.CopyFromDeviceToHost(vel_hv);
		Div_velw.CopyFromDeviceToHost(vel_hw);

		//DevcoefMatrix.cudaRelease();
		//DevRHS.cudaRelease();
		//Div_velu.cudaRelease();
		//Div_velv.cudaRelease();
		//Div_velw.cudaRelease();

	}


	/*-----------------------------------------------------------------------------*/
	/*----------------------------------Solve Pressure---------------------------- */
	/*-----------------------------------------------------------------------------*/
	//4、压力求解
	__global__ void K_Projection(Grid3f pressure, Grid3f buf, GridCoef coefMatrix, Grid3f RHS, Grid3f residual)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = pressure.nx;
		int ny = pressure.ny;
		int nz = pressure.nz;

		//if (i >= nx) return;
		//if (j >= ny) return;
		//if (k >= nz) return;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
		int k0 = coefMatrix.Index(i, j, k);
		Coef A_ijk = coefMatrix[k0];

		float a = A_ijk.a;
		float x0 = A_ijk.x0;
		float x1 = A_ijk.x1;
		float y0 = A_ijk.y0;
		float y1 = A_ijk.y1;
		float z0 = A_ijk.z0;
		float z1 = A_ijk.z1;
		float p_ijk;
		//// 			for (int it = 0; it < 1; it++)

		//	// 				buf[k0] = 0.0f;// pressure[k0];
		//	// 				__syncthreads();

		p_ijk = RHS[k0];
		if (i > 0) p_ijk += x0*buf(i - 1, j, k);
		if (i < nx - 1) p_ijk += x1*buf(i + 1, j, k);
		if (j > 0) p_ijk += y0*buf(i, j - 1, k);
		if (j < ny - 1) p_ijk += y1*buf(i, j + 1, k);
		if (k > 0) p_ijk += z0*buf(i, j, k - 1);
		if (k < nz - 1) p_ijk += z1*buf(i, j, k + 1);

		pressure[k0] = p_ijk / a;
		residual[k0] = p_ijk - a*buf(i, j, k);
		}

	}


	//4、压力求解
	void HybridMultiPhaseFluid::Projection(Grid3f pressure1, Grid3f Host_dataP,Grid3f buf, GridCoef coefMatrix, Grid3f RHS, int numIter,float dt)
	{		

		Projection_coefMatrix.CopyFromHostToDevice(coefMatrix);
		Projection_RHS.CopyFromHostToDevice(RHS);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		
		Grid3f host_residual;
		host_residual.SetSpace(pressure1.nx, pressure1.ny, pressure1.nz);
		Grid3f device_residual;
		device_residual.cudaSetSpace(pressure1.nx, pressure1.ny, pressure1.nz);

		//雅克比迭代求解压力
		Projection_pressure.cudaClear();
		for (int i = 0; i < numIter; i++)
		{
			device_residual.cudaClear();
			//K_CopyData << < dimGrid, dimBlock >> >(temp, Projection_buf);
			K_CopyData << < dimGrid, dimBlock >> >(Projection_buf, Projection_pressure);
			//K_CopyData << < dimGrid, dimBlock >> >(Projection_pressure, temp);
			//Projection_buf.Swap(Projection_pressure);
			//buf.Swap(pressure1);
			
			K_Projection << < dimGrid, dimBlock >> > (Projection_pressure, Projection_buf, Projection_coefMatrix, Projection_RHS, device_residual);
			cuSynchronize();

			device_residual.CopyFromDeviceToHost(host_residual);

			float totalRes = 0.0f;
			for (size_t j = 0; j < host_residual.elementCount; j++)
			{
				totalRes += abs(host_residual[j]);
			}
			std::cout << "Residual error at iter " << i << " : " << totalRes / host_residual.elementCount << std::endl;
		}
		Projection_pressure.CopyFromDeviceToHost(pressure1);
		//Projection_coefMatrix.cudaRelease();
		//Projection_RHS.cudaRelease();
		//Projection_buf.cudaRelease();
		//Projection_pressure.cudaRelease();

		host_residual.Release();
		device_residual.cudaRelease();
	}


	/*-----------------------------------------------------------------------------*/
	/*----------------------------------UpdateVelocity---------------------------- */
	/*-----------------------------------------------------------------------------*/
	__global__ void K_UpdateVelocity(Grid3f vel_u, Grid3f vel_v, Grid3f vel_w, Grid3f pressure, Grid3f mass, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = mass.nx;
		int ny = mass.ny;
		int nz = mass.nz;

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		if (i == 0) { vel_u(i, j, k) = 0.0f; return; }
		if (i == nx - 1) { vel_u(i + 1, j, k) = 0.0f; return; }
		if (j == 0) { vel_v(i, j, k) = 0.0f; return; }
		if (j == ny - 1) { vel_v(i, j + 1, k) = 0.0f; return; }
		if (k == 0) { vel_w(i, j, k) = 0.0f; return; }
		if (k == nz - 1) { vel_w(i, j, k + 1) = 0.0f; return; }
		int index;
		float c;
		if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
		{
			int nxy = nx*ny;
			//float h = pfParams.h;
			float h = 0.005f;

			index = mass.Index(i, j, k);
			c = 0.5f*(mass[index - 1] + mass[index]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			vel_u(i, j, k) -= dt*(pressure[index] - pressure[index - 1]) / h / (c*RHO1 + (1.0f - c)*RHO2);//(c*RHO1 + (1.0f - c)*RHO2)定义密度

			c = 0.5f*(mass[index] + mass[index - nx]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			vel_v(i, j, k) -= dt*(pressure[index] - pressure[index - nx]) / h / (c*RHO1 + (1.0f - c)*RHO2);

			c = 0.5f*(mass[index] + mass[index - nxy]);
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			vel_w(i, j, k) -= dt*(pressure[index] - pressure[index - nxy]) / h / (c*RHO1 + (1.0f - c)*RHO2);
		}
	}
	//5、速度更新
	void HybridMultiPhaseFluid::UpdateVelocity(Grid3f vel_u1, Grid3f vel_v1, Grid3f vel_w1, Grid3f pressure2, Grid3f mass_host, float dt)
	{

		vel_uu.CopyFromHostToDevice(vel_u1);
		vel_uv.CopyFromHostToDevice(vel_v1);
		vel_uw.CopyFromHostToDevice(vel_w1);
		Updatamass.CopyFromHostToDevice(mass_host);
		Updatapressure.CopyFromHostToDevice(pressure2);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, (nz + dimBlock.z - 1) / dimBlock.z);
		K_UpdateVelocity << < dimGrid, dimBlock >> > (vel_uu, vel_uv, vel_uw, Updatapressure, Updatamass,dt);
		cuSynchronize();
		vel_uu.CopyFromDeviceToHost(vel_u1);
		vel_uv.CopyFromDeviceToHost(vel_v1);
		vel_uw.CopyFromDeviceToHost(vel_w1);

		float totalDivergence = 0.0f;
		for (size_t i = 0; i < pressure2.nx; i++)
		{
			for (size_t j = 0; j < pressure2.ny; j++)
			{
				for (size_t k = 0; k < pressure2.nz; k++)
				{
					totalDivergence += abs((vel_u1(i+1, j, k)- vel_u1(i, j, k))+ (vel_v1(i, j+1, k) - vel_v1(i, j, k))+ (vel_w1(i, j, k+1) - vel_w1(i, j, k)));
				}
			}
		}

		cout << "-----------Total Divergence: " << totalDivergence << std::endl;

		//vel_uu.cudaRelease();
		//vel_uv.cudaRelease();
		//vel_uw.cudaRelease();
		//Updatamass.cudaRelease();
		//Updatapressure.cudaRelease();

	}

	//void HybridMultiPhaseFluid::Restriction(Grid3f src, Grid3f std)
	//{
	//	int nx = src.nx;
	//	int ny = src.ny;
	//	int nz = src.nz;
	//	for (int i = 0; i < nx/2-1; i++)
	//	{
	//		for (int j = 0; j < ny/2-1; j++)
	//		{
	//			for (int k = 0; k < nz/2; k++)
	//			{
	//				std(i, j, k) = (src(2 * i, j*2, k) / 4 + src(2 * i, 2 * j+1, k) / 2 + src(2 * i, 2 * j+2, k) / 4 + src(2 * i+1, 2 * j, k) / 2 + src(2 * i+1, 2 * j+1, k) + src(2 * i+1, 2 * j +2, k) / 2 + src(2 * i + 2, 2 * j, k) / 4 + src(2 * i + 2, 2 * j+1, k) / 2 + src(2 * i + 2, 2 * j + 2, k) / 4) / 4;
	//			}
	//		}
	//	}
	//}

	//void HybridMultiPhaseFluid::Interplation(Grid3f src, Grid3f std)
	//{
	//	int nx = src.nx;
	//	int ny = src.ny;
	//	int nz = src.nz;
	//	//四个角点

	//	std(0, 0, 0) = src(0, 0, 0) / 4;
	//	std(0, 2 * nx, 0) = src(0, nx - 1, 0) / 4;
	//	std(2 * ny, 0, 0) = src(ny - 1, 0, 0) / 4;
	//	std(2 * nx, 2 * ny, 0) = src(nx - 1, ny - 1, 0) / 4;
	//	//第一行
	//	for (int i = 0; i < 1; i++)
	//	{
	//		for (int j = 2; j <2*ny-1 ; j+=2)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = (src(i, j/ 2-1, k) + src(i, j/ 2, k)) / 4;
	//			}
	//		}

	//		for (int j = 1; j < 2*ny; j+=2)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = src(i, (j-1) / 2, k)/2;
	//			}
	//		}
	//	}
	//	//最后一行
	//	//黑点
	//	for (int i = 2*nx; i <= 2*nx; i++)
	//	{
	//		for (int j = 2; j < 2*nx-1; j+=2)
	//		{
	//			for (int k = 0; k <nz; k++)
	//			{
	//				std(i, j, k) = (src(i / 2-1, j/ 2-1, k) + src(i / 2-1, j / 2, k)) / 4;
	//			}
	//		}
	//	//绿点
	//		for (int j = 1; j < 2*ny; j+=2)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = src(i / 2-1, (j-1) / 2, k) / 2;
	//			}
	//		}
	//	}

	//	//第一列
	//	//黑点
	//	for (int i = 2; i < 2 * nx - 1; i += 2)
	//	{
	//		for (int j = 0; j < 1; j++)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = (src(i / 2 - 1, j, k) + src(i / 2 + 1, j, k) / 4);
	//			}
	//		}
	//	}
	//	//绿点
	//	for (int i = 1; i < 2 * nx; i += 2)
	//	{
	//		for (int j = 0; j < 1; j++)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = src((i-1)/ 2, j, k) / 2;
	//			}
	//		}
	//	}
	//	//最后一列
	//	//黑点
	//	for (int i = 2; i < 2 * nx; i += 2)
	//	{
	//		for (int j = 2 * ny; j <= 2 * ny; j++)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = src(i / 2-1, j / 2-1, k) + src(i / 2, j / 2-1, k) / 4;
	//			}
	//		}
	//	}
	//	//蓝点
	//	for (int i = 1; i < 2 * nx; i += 2)
	//	{
	//		for (int j = 2 * ny; j <= 2 * ny; j++)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = src((i-1) / 2, j / 2-1, k) / 2;
	//			}
	//		}
	//	}

	//	//处理中间部分的点
	//	//紫点
	//	for (int i = 1; i < 2*nx; i+=2)
	//	{
	//		for (size_t j = 1; j < 2*ny; j+=2)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = src(i-1 / 2, j-1 / 2, k);
	//			}
	//		}
	//	}
	//	//蓝点
	//	for (int i = 1; i < 2*nx; i+=2)
	//	{
	//		for (int j = 2; j < 2*ny-1; j+=2)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = (src((i-1)/ 2, j / 2-1, k) + src((i -1)/ 2, j/ 2, k)) / 2;
	//			}
	//		}
	//	}
	//	//绿点
	//	for (int i = 2; i < 2*nx-1; i+=2)
	//	{
	//		for (int j = 1; j < 2*ny; j+=2)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = (src(i/ 2-1, j-1 / 2, k) + src(i/ 2, j-1 / 2, k)) / 2;
	//			}
	//		}
	//	}
	//	//黑点
	//	for (int i = 2; i < 2*nx-1; i+=2)
	//	{
	//		for (int j = 2; j < 2*ny-1; j+=2)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				std(i, j, k) = (src(i/ 2-1,j/ 2-1, k) + src(i/ 2-1, j / 2, k) + src(i/ 2, j/ 2-1, k) + src(i/ 2, j/ 2, k)) / 4;
	//			}
	//		}
	//	}
	//}

	//void HybridMultiPhaseFluid::Jacobi(Grid3f pressure, Grid3f buf, GridCoef coefMatrix, Grid3f RHS)
	//{
	//	int nx = pressure.nx;
	//	int ny = pressure.ny;
	//	int nz = pressure.nz;

	//	for (int i = 1; i < nx-1; i++)
	//	{
	//		for (int j = 1; j < ny-1; j++)
	//		{
	//			for (int k = 1; k < nz-1; k++)
	//			{
	//				int k0 = coefMatrix.Index(i, j, k);
	//				Coef A_ijk = coefMatrix[k0];
	//				float a = A_ijk.a;
	//				float x0 = A_ijk.x0;
	//				float x1 = A_ijk.x1;
	//				float y0 = A_ijk.y0;
	//				float y1 = A_ijk.y1;
	//				float z0 = A_ijk.z0;
	//				float z1 = A_ijk.z1;
	//				pressure(i, j, k) += (x0*buf(i - 1, j, k) + x1*buf(i + 1, j, k) + y0*buf(i, j + 1, k) + y1*buf(i, j + 1, k) + z0*buf(i, j, k - 1) + z1*buf(i, j, k + 1) + RHS(i, j, k)) / a;
	//			}
	//		}
	//	}

	//}

	//void HybridMultiPhaseFluid::Correction(Grid3f r1, Grid3f b1, Grid3f A1, Grid3f u1)
	//{
	//	int nx = u1.nx;
	//	int ny = u1.ny;
	//	int nz = u1.nz;

	//	for (int i = 0; i < nx; i++)
	//	{
	//		for (int j = 0; j < ny; j++)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				r1(i, j, k) = b1(i, j, k) - A1(i, j, k)*u1(i, j, k);
	//			}
	//		}
	//	}
	//}
	//void HybridMultiPhaseFluid::CorrectionOnCarse(Grid3f u, Grid3f v)
	//{
	//	int nx = u.nx;
	//	int ny = u.ny;
	//	int nz = u.nz;

	//	for (int i = 0; i < nx; i++)
	//	{
	//		for (int j = 0; j < ny; j++)
	//		{
	//			for (int k = 0; k < nz; k++)
	//			{
	//				u(i, j, k) += v(i, j, k);
	//			}

	//		}

	//	}
	//}
	//void HybridMultiPhaseFluid::Multigrid(Grid3f A, Grid3f b, Grid3f u, int k)
	//{
	//	Grid3f buf;
	//	Grid3f r1;
	//	Grid3f V;
	//	Grid3f V1;
	//	buf = u;
	//	Jacobi(u, buf, coefMatrix, b);//前光滑化
	//	Correction(r1, b, A, u);//误差计算
	//	Restriction(b, r1);//限制操作，细网格到粗网格
	//	Multigrid(A, b, V, k - 1);//在粗网格上求解
	//	Interplation(V1, V);//插值操作，由粗网格到细网格
	//	CorrectionOnCarse(u, V1);//粗网格校正
	//	Jacobi(u, buf, coefMatrix, b);//后光滑化

	//}






	void HybridMultiPhaseFluid::Invoke(unsigned char type, unsigned char key, int x, int y)
	{
		cout << "HybridMultiPhaseFluid Key Pressed: " << key << endl;
		switch (type)
		{
		case 'K':
		{
			switch (key) {
			case 'u':
			{
				render_id++;
				render_id %= PHASE_SIZE;
				//vel_v(10, 10, 1) += 10.5f;

				ren_massfield = (massGrid_phase[0].data);
				ren_mass = (mass_phase[render_id]);
				ren_marker = (marker_phase[render_id].data);
			}

			break;

			default:
				break;
			}
		}
		break;

		default:
			break;
		}
	}







	HybridMultiPhaseFluid::~HybridMultiPhaseFluid(void)
	{
	}



	//Using--断点无用，注释报错
	void HybridMultiPhaseFluid::ComputeDensity()
	{
		//CalculateLiquidFraction();
		//ComputeAirDensityOnGrid();
		//ComputeLiquidDensityOnParticle();
		//	ComputeAdjustedLiquidDensityOnParticle();
	}

	//Using--断点无用，注释报错
	void HybridMultiPhaseFluid::ComputeNeighbors()
	{
		//ComputeLiquidNeighbors();
		//ComputeAirNeighbors();
	}



}


