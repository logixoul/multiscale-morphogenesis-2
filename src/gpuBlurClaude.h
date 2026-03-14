#pragma once
#include "precompiled.h"
#include "util.h"

namespace gpuBlurClaude {
	Array2D<float> singleblurLikeCinder(Array2D<float> src, ivec2 dstSize, float sigma, GLenum wrap = GL_CLAMP_TO_BORDER);
	gl::TextureRef singleblurLikeCinder(gl::TextureRef src, ivec2 dstSize, float sigma, GLenum wrap = GL_CLAMP_TO_BORDER);
	// the -1.0f downscaleSigma is a sentinel
	std::vector<gl::TextureRef> buildGaussianPyramid(gl::TextureRef const& src, float scalePerLevel = 0.5f, float downscaleSigma = -1.0f);
	gl::TextureRef blurWithInvKernel(gl::TextureRef const& src);
}
