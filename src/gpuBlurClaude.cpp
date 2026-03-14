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
	Array2D<float> singleblurLikeCinder(Array2D<float> src, ivec2 dstSize, float sigma, GLenum wrap) {
		return dl<float>(singleblurLikeCinder(gtex(src), dstSize, sigma, wrap));
	}
	gl::TextureRef singleblurLikeCinder(gl::TextureRef src, ivec2 dstSize, float sigma, GLenum wrap) {
		GPU_SCOPE("singleblur");
		float gaussW = sigma == -1.0f ? 4.0f : sigma;
		float hscale = float(dstSize.x) / src->getWidth();
		float vscale = float(dstSize.y) / src->getHeight();
		
		string shaderH =
			"int dstX = int(gl_FragCoord.x);"
			"int dstY = int(gl_FragCoord.y);"
			"float srcX = (dstX + 0.5f) / scaleX;"
			"float srcY = (dstY + 0.5f) / scaleY;"
			"float filterScaleX = max(1.0f, 1.0f / scaleX);"
			"float support = max(0.5f, filterScaleX * 1.25);"
			"float sum = 0.0;"
			"float wsum = 0.0;"
			"float cen = (dstX + .5f) / scaleX;"
			"int start = int(cen - support + 0.5f);"
			"int end = int(cen + support + 0.5f);"
			"for (int i = start; i < end; ++i) {"
			"    if (i < 0 || i >= int(texSize.x)) continue;"
			"	 float d = (float(i) + 0.5f - cen) / filterScaleX;"
			"	 float w = exp(-2.0f * d * d);"
			"    ivec2 p = ivec2(i, dstY);"
			"    sum += w * texelFetch(tex, p, 0).r;"
			"    wsum += w;"
			"}"
			"_out.r = sum / wsum;"
			;

		string shaderV =
			"int dstX = int(gl_FragCoord.x);"
			"int dstY = int(gl_FragCoord.y);"
			"float srcX = (dstX + 0.5f) / scaleX;"
			"float srcY = (dstY + 0.5f) / scaleY;"
			"float filterScaleY = max(1.0f, 1.0f / scaleY);"
			"float support = max(0.5f, filterScaleY * 1.25);"
			"float sum = 0.0;"
			"float wsum = 0.0;"
			"float cen = (dstY + .5f) / scaleY;"
			"int start = int(cen - support + 0.5f);"
			"int end = int(cen + support + 0.5f);"
			"for (int i = start; i < end; ++i) {"
			"    if (i < 0 || i >= int(texSize.y)) continue;"
			"	 float d = (float(i) + 0.5f - cen) / filterScaleY;"
			"	 float w = exp(-2.0f * d * d);"
			"    ivec2 p = ivec2(dstX, i);"
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
		auto hscaled = shade2(src, shaderH,
			ShadeOpts()
			.dstRectSize(ivec2(dstSize.x, src->getHeight()))
			.scale(hscale, 1.0f)
			.uniform("scaleX", hscale)
			.uniform("scaleY", 1.0f)
			.uniform("isHorizontal", 1)
			.uniform("sigma", gaussW)
		);
		setWrap(hscaled, wrap);
		if (wrap == GL_CLAMP_TO_BORDER) {
			setTextureBorderColor(hscaled, 0, 0, 0, 0);
		}
		auto vscaled = shade2(hscaled, shaderV,
			ShadeOpts()
			.dstRectSize(dstSize)
			.uniform("scaleX", 1.0f)
			.uniform("scaleY", vscale)
			.uniform("isHorizontal", 0)
			.uniform("sigma", gaussW));
		return vscaled;
	}

	std::vector<gl::TextureRef> buildGaussianPyramid(gl::TextureRef const& src, float scalePerLevel, float downscaleSigma) {
		std::vector<gl::TextureRef> result;
		result.push_back(src);
		auto state = src;
		while (true) {
			int minDim = std::min(state->getWidth(), state->getHeight());
			if(minDim <= 2)
				break;
			ivec2 dstSize = ivec2(state->getWidth() * scalePerLevel, state->getHeight() * scalePerLevel);
			state = singleblurLikeCinder(state, dstSize, downscaleSigma, GL_CLAMP_TO_EDGE);
			result.push_back(state);
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
