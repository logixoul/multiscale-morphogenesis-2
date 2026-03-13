#include "precompiled.h"
#include "gpuBlurClaude.h"
#include "gpuBlur2_5.h"
#include "gpgpu.h"
#include "stuff.h"

namespace gpuBlurClaude {
	std::vector<gl::TextureRef> buildGaussianPyramid(gl::TextureRef const& src, float scalePerLevel) {
		std::vector<gl::TextureRef> result;
		result.push_back(src);
		auto state = src;
		int minDim = std::min(src->getWidth(), src->getHeight());
		while (minDim > 2) {
			state = gpuBlur2_5::singleblur(state, scalePerLevel, scalePerLevel, GL_CLAMP_TO_EDGE);
			result.push_back(state);
			minDim *= scalePerLevel;
		}
		return result;
	}

	gl::TextureRef blurWithInvKernel(gl::TextureRef const& src) {
		// Build Gaussian pyramid. Each level is half the resolution of the previous.
		std::vector<gl::TextureRef> levels = buildGaussianPyramid(src, .5f);
		
		// 1/r kernel in 2D: each octave contributes equal weight,
		// so each pyramid level gets equal weight.
		int numLevels = (int)levels.size();
		float weight = 1.0f / numLevels;
		ivec2 dstSize = ivec2(src->getWidth(), src->getHeight());

		auto result = shade2(levels[0],
			"_out.rgb = fetch3() * _w;",
			ShadeOpts().uniform("_w", weight).dstRectSize(dstSize));

		for (int i = 1; i < numLevels; i++) {
			auto upscaled = gpuBlur2_5::upscale(levels[i], dstSize);
			result = shade2(result, upscaled,
				"_out.rgb = fetch3() + fetch3(tex2) * _w;",
				ShadeOpts().uniform("_w", weight));
		}

		return result;
	}

}
