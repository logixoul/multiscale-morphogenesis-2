#pragma once

#include "precompiled.h"
#include "Array2D_imageProc.h"
#include "gpgpu.h"

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

template<class T, class FetchFunc>
vec2 gradient_i_nodiv_sobel(Array2D<T>& src, ivec2 const& p)
{
	float gx =
		1.0f * FetchFunc::fetch(src, p.x + 1, p.y - 1)
		+ 2.0f * FetchFunc::fetch(src, p.x + 1, p.y)
		+ 1.0f * FetchFunc::fetch(src, p.x + 1, p.y + 1)
		- 1.0f * FetchFunc::fetch(src, p.x - 1, p.y - 1)
		- 2.0f * FetchFunc::fetch(src, p.x - 1, p.y)
		- 1.0f * FetchFunc::fetch(src, p.x - 1, p.y + 1);

	float gy =
		1.0f * FetchFunc::fetch(src, p.x - 1, p.y + 1)
		+ 2.0f * FetchFunc::fetch(src, p.x, p.y + 1)
		+ 1.0f * FetchFunc::fetch(src, p.x + 1, p.y + 1)
		- 1.0f * FetchFunc::fetch(src, p.x - 1, p.y - 1)
		- 2.0f * FetchFunc::fetch(src, p.x, p.y - 1)
		- 1.0f * FetchFunc::fetch(src, p.x + 1, p.y - 1);

	return vec2(gx, gy);
}

template<class T, class FetchFunc>
Array2D<vec2> get_gradients_sobel(Array2D<T>& src)
{
	auto src2 = src.clone();
	forxy(src2)
		src2(p) /= 2.0f;

	Array2D<vec2> gradients(src.w, src.h);

	for (int x = 0; x < src.w; x++)
	{
		gradients(x, 0) = gradient_i_nodiv_sobel<T, FetchFunc>(src2, ivec2(x, 0));
		gradients(x, src.h - 1) = gradient_i_nodiv_sobel<T, FetchFunc>(src2, ivec2(x, src.h - 1));
	}
	for (int y = 1; y < src.h - 1; y++)
	{
		gradients(0, y) = gradient_i_nodiv_sobel<T, FetchFunc>(src2, ivec2(0, y));
		gradients(src.w - 1, y) = gradient_i_nodiv_sobel<T, FetchFunc>(src2, ivec2(src.w - 1, y));
	}
	for (int y = 1; y < src.h - 1; y++) {
		for (int x = 1; x < src.w - 1; x++) {
			gradients(x, y) = gradient_i_nodiv_sobel<T, WrapModes::NoWrap>(src2, ivec2(x, y));
		}
	}

	return gradients;
}

template<class T>
Array2D<vec2> get_gradients_sobel(Array2D<T> src)
{
	return get_gradients_sobel<T, WrapModes::DefaultImpl>(src);
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

