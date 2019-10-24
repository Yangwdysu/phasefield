#include<glfw3.h>
#include<algorithm>
#include<cmath>
#include<cstring>
#include<iostream>
#include<fstream>

using namespace std;
enum Cell{EMPY,FLUID,SOLID};
const int N = 32;
const int Np1 = N + 1;
double u[2 * N*(N + 1)];
double mu[2 * N*(N + 1)];
double v[2 * N*(N + 1)];
double mv[2 * N*(N + 1)];
double ux[2 * N*(N + 1)];
double p[N*N];
double div[N*N];
Cell flags[N*N];
const int PN = (N - 2) / 4 * (N - 4) * 4;
double px[PN], py[PN], pvx[PN], pvy[PN];
double dt = 0.001;

void init()
{
	int idx = 0;
	memset(pvx, 0, sizeof(pvx));
	memset(pvy, 0, sizeof(pvy));
	for (int j = 1; j < N-1; j++)
	{
		for (int i = 1; i < N-1; i++)
		{
			if (idx<PN&&i-1<(N-2)/4)
			{
				for (int jj = 0; jj < 2; jj++)
				{
					for (int ii = 0; ii < 2; ii++)
					{
						px[idx] = i + 0.25 + ii*0.5;
						py[idx] = j + 0.25 + jj*0.5;
					}
				}
			}
		}
	}
}
void particle2grid() 
{
	memset(u, 0, sizeof(u));
	memset(v, 0, sizeof(v));
	memset(mu, 0, sizeof(mu));
	for (int j=0; j < N; j++)
		for (int i = 0; i < 32; i++)
		{
			flags[i + j*N] =(i == 0 || j == 0 || i == (N - 1) || j = (N - 1)) ? SOLID : EMPY;
		}
	for (int k = 0; k < PN; k++)
	{
		int i(px[k]), j(py[k]), fi(px[k] - 0.5), fj(py[k] - 0.5);
		flags[i + j*N] = FLUID;

		for (int jj = (fj); jj < (fj)+2; jj++)
		{
			for (int ii = (i); ii < (i)+2; ii++)
			{
				int idex = ii + jj*Np1;
				u[idex] += pvx[k] * (1 - fabs(ii - px[k]))*(1 - fabs(jj + 0.5 - py[k]));
				mu[idex] += (1 - fabs(ii - px[k]))*(1 - fabs(jj + 0.5 - py[k]));
			}
		}
		for (int jj = (j); jj < (j)+2; jj++)
		{
			for (int ii = (fi); ii < fi + 2; ii++)
			{
				int idex = ii + jj*Np1;
				v[idex] += pvy[k] * (1 - fabs(ii + 0.5 - px[k]))*(1 - fabs(jj - py[k]));
				mv[idex] += (1 - fabs(ii + 0.5 - px[k]))*(1 - fabs(jj - py[k]));
			}
		}

		for (int k = 0; k < N*Np1; k++)
		{
			if (mu[k] > 1e - 8)
			{
				mu[k] = 1 / mu[k];
				u[k] *= mu[k];
			}
		}
	}
}
int main()
{

}



