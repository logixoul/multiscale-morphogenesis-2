#include "precompiled.h"
#include "gpuBlurClaude.h"
#include "gpuBlur2_5.h"
#include "gpgpu.h"
#include "stuff.h"

namespace gpuBlurClaude {
	// todo: move this to stuff.cpp/h. Copy-pasted in gpuBlur2_5.cpp as well.
	void setTextureBorderColor(gl::TextureRef tex, float r, float g, float b, float a) {
		bind(tex);
		float color[] = { r, g, b, a };
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);
	}
	static gl::TextureRef singleblurLikeCinder(gl::TextureRef src, float hscale, float vscale, float sigma, GLenum wrap) {
		GPU_SCOPE("singleblur");
		float gaussW = sigma == -1.0f ? 4.0f : sigma;
		
		string shader =
			"int x = int(gl_FragCoord.x);"
			"int y = int(gl_FragCoord.y);"
			"float srcX = (x + 0.5f) * scaleX - 0.5f;"
			"float srcY = (y + 0.5f) * scaleY - 0.5f;"
			"float srcPos = (isHorizontal != 0) ? srcX : srcY;"
			"int srcLen = (isHorizontal != 0) ? int(texSize.x) : int(texSize.y);"
			"float sum = 0.0;"
			"float wsum = 0.0;"
			"for (int k = -2; k <= 2; ++k) {"
			"    int i = int(floor(srcPos + float(k) + 0.5));"
			"    if (i < 0 || i >= srcLen) continue;"
			"    float d = (float(i) - srcPos) / sigma;"
			"    float w = exp(-0.5 * d * d);"
			"    ivec2 p = (isHorizontal != 0) ? ivec2(i, y) : ivec2(x, i);"
			"    sum += w * texelFetch(tex, p, 0).r;"
			"    wsum += w;"
			"}"
			"_out.r = sum / wsum;"
			;

		setWrap(src, wrap);
		if (wrap == GL_CLAMP_TO_BORDER) {
			setTextureBorderColor(src, 0, 0, 0, 0);
		}
		//setWrapBlack(src);
		auto hscaled = shade2(src, shader,
			ShadeOpts()
			.scale(hscale, 1.0f)
			.uniform("scaleX", 1.0f / hscale)
			.uniform("scaleY", 1.0f)
			.uniform("isHorizontal", 1)
			.uniform("sigma", gaussW)
		);
		setWrap(hscaled, wrap);
		if (wrap == GL_CLAMP_TO_BORDER) {
			setTextureBorderColor(hscaled, 0, 0, 0, 0);
		}
		auto vscaled = shade2(hscaled, shader,
			ShadeOpts()
			.scale(1.0f, vscale)
			.uniform("scaleX", 1.0f)
			.uniform("scaleY", 1.0f / vscale)
			.uniform("isHorizontal", 0)
			.uniform("sigma", gaussW));
		return vscaled;
	}

	std::vector<gl::TextureRef> buildGaussianPyramid(gl::TextureRef const& src, float scalePerLevel, float downscaleSigma) {
		std::vector<gl::TextureRef> result;
		result.push_back(src);
		auto state = src;
		int minDim = std::min(src->getWidth(), src->getHeight());
		while (minDim > 2) {
			state = singleblurLikeCinder(state, scalePerLevel, scalePerLevel, downscaleSigma, GL_CLAMP_TO_EDGE);
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
