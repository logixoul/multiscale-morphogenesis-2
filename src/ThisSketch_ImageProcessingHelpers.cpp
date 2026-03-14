#include "precompiled.h"
#include "ThisSketch_ImageProcessingHelpers.h"
#include "stuff.h"
#include "gpuBlurClaude.h"

namespace ThisSketch {

	vec2 perpLeft(vec2 const& v) {
		return vec2(-v.y, v.x);
	}

	Img subtract(Img const& a, Img const& b) {
		Img result = a.clone();
		for (int i = 0; i < result.area; i++) {
			result.data[i] -= b.data[i];
		}
		return result;
	}

	Img add(Img const& a, Img const& b) {
		Img result = a.clone();
		for (int i = 0; i < result.area; i++) {
			result.data[i] += b.data[i];
		}
		return result;
	}

	Img multiply(Img const& a, Img const& b) {
		Img result = a.clone();
		for (int i = 0; i < result.area; i++) {
			result.data[i] *= b.data[i];
		}
		return result;
	}

	Img multiply(Img const& a, float scalar) {
		Img result = a.clone();
		for (int i = 0; i < result.area; i++) {
			result.data[i] *= scalar;
		}
		return result;
	}

	float mulContrastize(float i, float contrastizeStrength) {
		i = ci::constrain(i, 0.0f, 1.0f);
		const bool invert = i > .5f;
		if (invert) {
			i = 1.0f - i;
		}
		i *= 2.0f;
		i = pow(i, contrastizeStrength);
		i *= .5f;
		if (invert) {
			i = 1.0f - i;
		}
		return i;
	}

	Array2D<float> resize(Array2D<float> src, ivec2 dstSize, const ci::FilterBase& filter)
	{
		ci::ChannelT<float> tmpSurface(
			src.w, src.h, /*rowBytes*/sizeof(float) * src.w, 1, src.data);
		ci::ChannelT<float> resizedSurface(dstSize.x, dstSize.y);
		ci::ip::resize(tmpSurface, &resizedSurface, filter);
		Array2D<float> resultArray = resizedSurface;
		return resultArray;
	}

	/*std::vector<Img> buildGaussianPyramid(Img src, float scalePerLevel) {
		auto tex = gtex(src);
		std::vector<gl::TextureRef> scales = gpuBlurClaude::buildGaussianPyramid(tex, scalePerLevel);
		std::vector<Img> result;
		for (auto& scale : scales) {
			result.push_back(dl<float>(scale));
		}
		return result;
	}*/

	std::vector<Img> buildGaussianPyramid(Img src, float scalePerLevel) {
		std::vector<Img> scales;
		auto state = src.clone();
		static const auto filter = ci::FilterGaussian();
		while (true)
		{
			const int size = std::min(state.w, state.h);
			if (size <= 2)
				break;
			scales.push_back(state);
			ivec2 newSize = ivec2(vec2(state.Size()) * scalePerLevel);
			state = ThisSketch::resizeGaussianCpuSimple2Trimmed(state, newSize);
		}
		return scales;
	}

	Array2D<float> resizeGaussianCpuSimple(Array2D<float> const& src, ivec2 dstSize)
	{
		auto clampi = [](int v, int lo, int hi, bool* outOfBounds) {
			*outOfBounds = (v < lo) || (v > hi);
			return std::max(lo, std::min(v, hi));
			};

		const float scaleX = float(src.w) / float(dstSize.x);
		const float scaleY = float(src.h) / float(dstSize.y);

		const int radiusX = 4;
		const int radiusY = 4;

		ci::FilterGaussian filter;

		Array2D<float> tmp(dstSize.x, src.h);
		Array2D<float> out(dstSize.x, dstSize.y);

		// Horizontal pass: src -> tmp
		for (int dstY = 0; dstY < src.h; ++dstY) {
			for (int dstX = 0; dstX < dstSize.x; ++dstX) {
				const float srcX = (dstX + 0.5f) * scaleX - 0.5f;
				const float cen = (dstX + .5f) * scaleX;

				float sum = 0.0f;
				float wsum = 0.0f;
				for (int i = -radiusX; i <= radiusX; ++i) {
					bool outOfBounds;
					const int ix = clampi(int(srcX + i + 0.5f), 0, src.w - 1, &outOfBounds);
					float d = (ix + 0.5f - cen);
					const float w = std::exp(-2.0f * d * d);
					if (!outOfBounds) {
						sum += w * src.data[dstY * src.w + ix];
						wsum += w;
					}
				}
				tmp.data[dstY * tmp.w + dstX] = sum / wsum;
			}
		}

		// Vertical pass: tmp -> out
		for (int dstY = 0; dstY < dstSize.y; ++dstY) {
			const float srcY = (dstY + 0.5f) * scaleY - 0.5f;
			const float cen = (dstY + .5f) * scaleY;

			for (int dstX = 0; dstX < dstSize.x; ++dstX) {
				float sum = 0.0f;
				float wsum = 0.0f;
				for (int i = -radiusY; i <= radiusY; ++i) {
					bool outOfBounds;
					const int iy = clampi(int(srcY + i + 0.5f), 0, src.h - 1, &outOfBounds);
					float d = (iy + 0.5f - cen);
					const float w = std::exp(-2.0f * d * d);
					if (!outOfBounds) {
						sum += w * tmp.data[iy * tmp.w + dstX];
						wsum += w;
					}
				}
				out.data[dstY * out.w + dstX] = sum / wsum;
			}
		}

		return out;
	}

	class FilterGaussianNonNormalized {
	public:
		float operator()(float x) const {
			return exp(-2.0f * x * x);
		}

		float getSupport() const { return 1.25f; }
	};

	Array2D<float> resizeGaussianCpuSimple2Trimmed(Array2D<float> const& src, ivec2 dstSize)
	{
		const int srcW = src.w;
		const int srcH = src.h;
		const int dstW = dstSize.x;
		const int dstH = dstSize.y;

		const float sx = dstW / (float)srcW;
		const float sy = dstH / (float)srcH;

		FilterGaussianNonNormalized filter;

		const float filterScaleX = std::max(1.0f, 1.0f / sx);
		const float filterScaleY = std::max(1.0f, 1.0f / sy);
		const float supportX = std::max(0.5f, filterScaleX * filter.getSupport());
		const float supportY = std::max(0.5f, filterScaleY * filter.getSupport());

		Array2D<float> tmp(dstW, srcH);
		Array2D<float> out(dstW, dstH);

		for (int dstY = 0; dstY < srcH; ++dstY) {
			for (int dstX = 0; dstX < dstW; ++dstX) {
				const float center = (dstX + .5f) / sx;
				int start = (int)(center - supportX + 0.5f);
				int end = (int)(center + supportX + 0.5f);
				if (start < 0) start = 0;
				if (end > srcW) end = srcW;

				float den = 0.0f;
				float sum = 0.0f;
				for (int i = start; i < end; ++i) {
					float d = (i + 0.5f - center) / filterScaleX;
					float w = filter(d);
					sum += w * src.data[dstY * srcW + i];
					den += w;
				}

				tmp.data[dstY * dstW + dstX] = sum / den;
			}
		}

		for (int dstY = 0; dstY < dstH; ++dstY) {
			const float center = (dstY + .5f) / sy;
			int start = (int)(center - supportY + 0.5f);
			int end = (int)(center + supportY + 0.5f);
			if (start < 0) start = 0;
			if (end > srcH) end = srcH;

			
			for (int dstX = 0; dstX < dstW; ++dstX) {
				float sum = 0.0f;
				float den = 0.0f;
				for (int i = start; i < end; ++i) {
					float d = (i + 0.5f - center) / filterScaleY;
					float w = filter(d);
					sum += w * tmp.data[i * dstW + dstX];
					den += w;
				}

				out.data[dstY * dstW + dstX] = sum / den;
			}
		}

		return out;
	}


	Array2D<float> resizeGaussianCpuSimple2(Array2D<float> const& src, ivec2 dstSize)
	{
		const int srcW = src.w;
		const int srcH = src.h;
		const int dstW = dstSize.x;
		const int dstH = dstSize.y;

		const float sx = dstW / (float)srcW;
		const float sy = dstH / (float)srcH;

		ci::FilterGaussian filter;

		const float filterScaleX = std::max(1.0f, 1.0f / sx);
		const float filterScaleY = std::max(1.0f, 1.0f / sy);
		const float supportX = std::max(0.5f, filterScaleX * filter.getSupport());
		const float supportY = std::max(0.5f, filterScaleY * filter.getSupport());

		Array2D<float> tmp(dstW, srcH);
		Array2D<float> out(dstW, dstH);

		for (int dstY = 0; dstY < srcH; ++dstY) {
			for (int dstX = 0; dstX < dstW; ++dstX) {
				const float cen = (dstX + .5f) / sx;
				int start = (int)(cen - supportX + 0.5f);
				int end = (int)(cen + supportX + 0.5f);
				if (start < 0) start = 0;
				if (end > srcW) end = srcW;

				float den = 0.0f;
				for (int i = start; i < end; ++i) {
					den += filter((i + 0.5f - cen) / filterScaleX);
				}

				const float sc = (den == 0.0f) ? 1.0f : (1.0f / den);
				float sum = 0.0f;
				float wsum = 0.0f;
				for (int i = start; i < end; ++i) {
					float w = sc * filter((i + 0.5f - cen) / filterScaleX);
					sum += w * src.data[dstY * srcW + i];
					wsum += w;
				}

				if (wsum == 0.0f) {
					int mid = (start + end) >> 1;
					if (mid < 0) mid = 0;
					if (mid >= srcW) mid = srcW - 1;
					tmp.data[dstY * dstW + dstX] = src.data[dstY * srcW + mid];
				}
				else {
					int ic = (int)(cen + 0.5f);
					if (ic < start) ic = start;
					else if (ic >= end) ic = end - 1;
					tmp.data[dstY * dstW + dstX] = sum + (1.0f - wsum) * src.data[dstY * srcW + ic];
				}
			}
		}

		for (int dstY = 0; dstY < dstH; ++dstY) {
			const float cen = (dstY + .5f) / sy;
			int start = (int)(cen - supportY + 0.5f);
			int end = (int)(cen + supportY + 0.5f);
			if (start < 0) start = 0;
			if (end > srcH) end = srcH;

			float den = 0.0f;
			for (int i = start; i < end; ++i) {
				den += filter((i + 0.5f - cen) / filterScaleY);
			}

			const float sc = (den == 0.0f) ? 1.0f : (1.0f / den);
			for (int dstX = 0; dstX < dstW; ++dstX) {
				float sum = 0.0f;
				float wsum = 0.0f;
				for (int i = start; i < end; ++i) {
					float w = sc * filter((i + 0.5f - cen) / filterScaleY);
					sum += w * tmp.data[i * dstW + dstX];
					wsum += w;
				}

				if (wsum == 0.0f) {
					int mid = (start + end) >> 1;
					if (mid < 0) mid = 0;
					if (mid >= srcH) mid = srcH - 1;
					out.data[dstY * dstW + dstX] = tmp.data[mid * dstW + dstX];
				}
				else {
					int ic = (int)(cen + 0.5f);
					if (ic < start) ic = start;
					else if (ic >= end) ic = end - 1;
					out.data[dstY * dstW + dstX] = sum + (1.0f - wsum) * tmp.data[ic * dstW + dstX];
				}
			}
		}

		return out;
	}



	gl::TextureRef redToLuminance(gl::TextureRef const& in) {
		return shade2(in,
			"_out.rgb = vec3(fetch1());",
			ShadeOpts().ifmt(GL_RGBA16F)
		);
	}

	float blendHardLight(float base, float blend) {
		if (blend < 0.5f) {
			return 2.0f * base * blend;
		}
		else {
			return 1.0f - 2.0f * (1.0f - base) * (1.0f - blend);
		}
	}

	

}