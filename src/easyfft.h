#pragma once
#include "util.h"
typedef std::complex<float> Complexf;

Array2D<Complexf> fft(Array2D<float> in, int flags);
Array2D<float> ifft(Array2D<Complexf> in, int flags);