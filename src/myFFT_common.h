#pragma once

#include "precompiled.h"

typedef float FFTScalar;
typedef vec2 FFTComplex;
enum class FFTDir { Forward, Backward };

void getAAndB(int N, vector<FFTComplex>& A, vector<FFTComplex>& B, FFTDir dir);