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
		
		float w0 = gpuBlur2_5::gauss(0.0, gaussW);
		float w1 = gpuBlur2_5::gauss(1.0, gaussW);
		float w2 = gpuBlur2_5::gauss(2.0, gaussW);
		float sum = 2.0f * (w1 + w2) + w0;
		w2 /= sum;
		w1 /= sum;
		w0 /= sum;
		stringstream weights;
		weights << fixed << "float w0=" << w0 << ", w1=" << w1 << ", w2=" << w2 << ";" << endl;
		
		string shader =
			"int x = int(gl_FragCoord.x);"
			"int y = int(gl_FragCoord.y);"
			"float srcX = (x + 0.5f) * scaleX - 0.5f;"
			"float srcY = (y + 0.5f) * scaleY - 0.5f;"
			"vec2 srcPos = vec2(srcX, srcY);"
			"vec2 offset = vec2(GB2_offsetX, GB2_offsetY);"
			"vec3 a[5];"
			"for (int i = -2; i <= 2; i++) {"
			"	ivec2 fetchPos = ivec2(srcPos + float(i) * offset + vec2(.5));"
			"	a[i+2] = texelFetch(tex, fetchPos, 0).rgb;"
			"}"
			+ weights.str() +
			"_out.rgb = w2 * (a[0] + a[4]) + w1 * (a[1] + a[3]) + w0 * a[2];";

		setWrap(src, wrap);
		if (wrap == GL_CLAMP_TO_BORDER) {
			setTextureBorderColor(src, 0, 0, 0, 0);
		}
		//setWrapBlack(src);
		auto hscaled = shade2(src, shader,
			ShadeOpts()
			.scale(hscale, 1.0f)
			.uniform("scaleX", hscale)
			.uniform("scaleY", 1.0f)
			.uniform("GB2_offsetX", 1.0f)
			.uniform("GB2_offsetY", 0.0f)
		);
		setWrap(hscaled, wrap);
		if (wrap == GL_CLAMP_TO_BORDER) {
			setTextureBorderColor(src, 0, 0, 0, 0);
		}
		auto vscaled = shade2(hscaled, shader,
			ShadeOpts()
			.uniform("GB2_offsetX", 0.0f)
			.uniform("GB2_offsetY", 1.0f)
			.scale(1.0f, vscale)
			.uniform("scaleX", 1.0f)
			.uniform("scaleY", vscale));
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
