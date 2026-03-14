#pragma once
#include "precompiled.h"

namespace gpuBlurClaude {
	// the -1.0f downscaleSigma is a sentinel
	std::vector<gl::TextureRef> buildGaussianPyramid(gl::TextureRef const& src, float scalePerLevel = 0.5f, float downscaleSigma = -1.0f);
	gl::TextureRef blurWithInvKernel(gl::TextureRef const& src);
}
