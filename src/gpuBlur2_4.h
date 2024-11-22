#pragma once
#include "precompiled.h"

namespace gpuBlur2_4 {
	gl::TextureRef run(gl::TextureRef src, int lvls);
	gl::TextureRef run_longtail(gl::TextureRef src, int lvls, float lvlmul);
	float getGaussW();
	float gauss(float f, float width);
	gl::TextureRef upscale(gl::TextureRef src, ci::ivec2 toSize);
	gl::TextureRef upscale(gl::TextureRef src, float hscale, float vscale);
	gl::TextureRef singleblur(gl::TextureRef src, float hscale, float vscale);
}