#pragma once

#include "precompiled.h"
#include "Array2D_imageProc.h"
#include "gpgpu.h"

namespace ThisSketch {

	using Img = Array2D<float>;

	vec2 perpLeft(vec2 const& v);

	Img subtract(Img const& a, Img const& b);
	Img add(Img const& a, Img const& b);
	Img multiply(Img const& a, Img const& b);
	Img multiply(Img const& a, float scalar);

	template<class T, class FetchFunc>
	Array2D<T> gaussianBlur3x3(Array2D<T> src) {
		T zero = ::zero<T>();
		Array2D<T> dst1(src.w, src.h);
		Array2D<T> dst2(src.w, src.h);
		forxy(dst1)
			dst1(p) = .25f * (2 * FetchFunc::fetch(src, p.x, p.y) + FetchFunc::fetch(src, p.x - 1, p.y) + FetchFunc::fetch(src, p.x + 1, p.y));
		forxy(dst2)
			dst2(p) = .25f * (2 * FetchFunc::fetch(dst1, p.x, p.y) + FetchFunc::fetch(dst1, p.x, p.y - 1) + FetchFunc::fetch(dst1, p.x, p.y + 1));
		return dst2;
	}

	float mulContrastize(float i, float contrastizeStrength);

	template<class T, class FetchFunc>
	float hessianDirectionalSecondDeriv(Array2D<T>& src, ivec2 const& p, vec2 const& d) {
		float fxx = FetchFunc::fetch(src, p.x + 1, p.y) - 2.0f * FetchFunc::fetch(src, p.x, p.y) + FetchFunc::fetch(src, p.x - 1, p.y);
		float fyy = FetchFunc::fetch(src, p.x, p.y + 1) - 2.0f * FetchFunc::fetch(src, p.x, p.y) + FetchFunc::fetch(src, p.x, p.y - 1);
		float fxy = 0.25f * (
			FetchFunc::fetch(src, p.x + 1, p.y + 1)
			- FetchFunc::fetch(src, p.x - 1, p.y + 1)
			- FetchFunc::fetch(src, p.x + 1, p.y - 1)
			+ FetchFunc::fetch(src, p.x - 1, p.y - 1));
		return d.x * d.x * fxx + 2.0f * d.x * d.y * fxy + d.y * d.y * fyy;
	}

	Array2D<float> resize(Array2D<float> src, ivec2 dstSize, const ci::FilterBase& filter);
	std::vector<Img> buildGaussianPyramid(Img src, float scalePerLevel = 0.5f);
	
	gl::TextureRef redToLuminance(gl::TextureRef const& in);
	float blendHardLight(float base, float blend);

	Array2D<float> resizeGaussianCpuSimple(Array2D<float> src, ivec2 dstSize);
	Array2D<float> resizeGaussianCpuSimple2(Array2D<float> src, ivec2 dstSize);

	Array2D<float> resizeGaussianCpuSimple2Trimmed(Array2D<float> src, ivec2 dstSize);

} // namespace ThisSketch