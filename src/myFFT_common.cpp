#include "precompiled.h"
#include "myFFT_common.h"

void getAAndB(int N, vector<FFTComplex>& A, vector<FFTComplex>& B, FFTDir dir) {
	A.clear();
	B.clear();
	A.resize(N / 2);
	B.resize(N / 2);
	for (int k = 0; k < N / 2; k++)
	{
		double kd = k;
		if (dir == FFTDir::Forward) {
			A[k].x = 1.0 - sin(M_PI * kd / (double)(N / 2));
			A[k].y = -1.0 * cos(M_PI * kd / (double)(N / 2));
			B[k].x = 1.0 + sin(M_PI * kd / (double)(N / 2));
			B[k].y = 1.0 * cos(M_PI * kd / (double)(N / 2));
		}
		else {
			A[k].x = 1.0 - sin(M_PI * kd / (double)(N / 2));
			A[k].y = 1.0 * cos(M_PI * kd / (double)(N / 2));
			B[k].x = 1.0 + sin(M_PI * kd / (double)(N / 2));
			B[k].y = -1.0 * cos(M_PI * kd / (double)(N / 2));
		}
		A[k] *= .5f;
		B[k] *= .5f;
	}
}

