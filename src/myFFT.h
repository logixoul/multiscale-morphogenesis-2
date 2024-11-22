#pragma once
#include "precompiled.h"
#include "myFFT_common.h"

class myFFT
{
public:
	static void fftGpu(gl::TextureRef& in);
	static void ifftGpu(gl::TextureRef& in, ivec2 returnSize);
};

