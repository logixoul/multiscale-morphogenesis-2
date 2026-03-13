#pragma once
#include "precompiled.h"

namespace gpuBlurClaude {
	std::vector<gl::TextureRef> buildGaussianPyramid(gl::TextureRef const& src, float scalePerLevel = 0.5f);
	gl::TextureRef blurWithInvKernel(gl::TextureRef const& src);
}
