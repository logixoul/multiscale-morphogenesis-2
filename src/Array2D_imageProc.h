#pragma once

#include "util.h"

template<class T>
ivec2 wrapPoint(Array2D<T> const& src, ivec2 p)
{
	ivec2 wp = p;
	wp.x %= src.w; if (wp.x < 0) wp.x += src.w;
	wp.y %= src.h; if (wp.y < 0) wp.y += src.h;
	return wp;
}
template<class T>
T const& getWrapped(Array2D<T> const& src, ivec2 p)
{
	return src(wrapPoint(src, p));
}
template<class T>
T const& getWrapped(Array2D<T> const& src, int x, int y)
{
	return getWrapped(src, ivec2(x, y));
}
template<class T>
T& getWrapped(Array2D<T>& src, ivec2 p)
{
	return src(wrapPoint(src, p));
}
template<class T>
T& getWrapped(Array2D<T>& src, int x, int y)
{
	return getWrapped(src, ivec2(x, y));
}
struct WrapModes {
	struct GetWrapped {
		template<class T>
		static T& fetch(Array2D<T>& src, int x, int y)
		{
			return ::getWrapped(src, x, y);
		}
	};
	struct Get_WrapZeros {
		template<class T>
		static T& fetch(Array2D<T>& src, int x, int y)
		{
			return ::get_wrapZeros(src, x, y);
		}
	};
	struct NoWrap {
		template<class T>
		static T& fetch(Array2D<T>& src, int x, int y)
		{
			return src(x, y);
		}
	};
	struct GetClamped {
		template<class T>
		static T& fetch(Array2D<T>& src, int x, int y)
		{
			return ::get_clamped(src, x, y);
		}
	};
	typedef GetWrapped DefaultImpl;
};
template<class T>
void aaPoint_i(Array2D<T>& dst, ivec2 p, T value)
{
	dst.wr(p) += value;
}
template<class T>
void aaPoint_i(Array2D<T>& dst, int x, int y, T value)
{
	dst.wr(x, y) += value;
}

template<class T, class FetchFunc>
void aaPoint(Array2D<T>& dst, vec2 p, T value)
{
	aaPoint<T, FetchFunc>(dst, p.x, p.y, value);
}
template<class T, class FetchFunc>
void aaPoint(Array2D<T>& dst, float x, float y, T value)
{
	int ix = x, iy = y;
	float fx = ix, fy = iy;
	if (x < 0.0f && fx != x) { fx--; ix--; }
	if (y < 0.0f && fy != y) { fy--; iy--; }
	float fractx = x - fx;
	float fracty = y - fy;
	float fractx1 = 1.0 - fractx;
	float fracty1 = 1.0 - fracty;
	FetchFunc::fetch(dst, ix, iy) += (fractx1 * fracty1) * value;
	FetchFunc::fetch(dst, ix, iy + 1) += (fractx1 * fracty) * value;
	FetchFunc::fetch(dst, ix + 1, iy) += (fractx * fracty1) * value;
	FetchFunc::fetch(dst, ix + 1, iy + 1) += (fractx * fracty) * value;
}
template<class T>
void aaPoint(Array2D<T>& dst, float x, float y, T value)
{
	aaPoint<T, WrapModes::DefaultImpl>(dst, p.x, p.y, value);
}
template<class T>
void aaPoint(Array2D<T>& dst, vec2 p, T value)
{
	aaPoint<T, WrapModes::DefaultImpl>(dst, p, value);
}
template<class T, class FetchFunc>
T getBilinear(Array2D<T> src, vec2 p)
{
	return getBilinear<T, FetchFunc>(src, p.x, p.y);
}
template<class T, class FetchFunc>
T getBilinear(Array2D<T> src, float x, float y)
{
	int ix = x, iy = y;
	float fx = ix, fy = iy;
	if (x < 0.0f && fx != x) { fx--; ix--; }
	if (y < 0.0f && fy != y) { fy--; iy--; }
	float fractx = x - fx;
	float fracty = y - fy;
	return lerp(
		lerp(FetchFunc::fetch(src, ix, iy), FetchFunc::fetch(src, ix + 1, iy), fractx),
		lerp(FetchFunc::fetch(src, ix, iy + 1), FetchFunc::fetch(src, ix + 1, iy + 1), fractx),
		fracty);
}
template<class T>
T getBilinear(Array2D<T> src, float x, float y)
{
	return getBilinear<T, WrapModes::DefaultImpl>(src, x, y);
}
template<class T>
T getBilinear(Array2D<T> src, vec2 p)
{
	return getBilinear<T, WrapModes::DefaultImpl>(src, p);
}

Array2D<float> to01(Array2D<float> a);
Array2D<vec3> to01(Array2D<vec3> a, float min, float max);
template<class T>
T& zero() {
	static T val = T()*0.0f;
	val = T()*0.0f;
	return val;
}

ivec2 clampPoint(ivec2 p, int w, int h);
template<class T>
T& get_clamped(Array2D<T>& src, int x, int y)
{
	return src(clampPoint(ivec2(x, y), src.w, src.h));
}
template<class T>
T const& get_clamped(Array2D<T> const& src, int x, int y)
{
	return src(clampPoint(ivec2(x, y), src.w, src.h));
}

template<class T>
Array2D<T> gauss3(Array2D<T> src) {
	T zero = ::zero<T>();
	Array2D<T> dst1(src.w, src.h);
	Array2D<T> dst2(src.w, src.h);
	forxy(dst1)
		dst1(p) = .25f * (2.0f * get_clamped(src, p.x, p.y) + get_clamped(src, p.x - 1, p.y) + get_clamped(src, p.x + 1, p.y));
	forxy(dst2)
		dst2(p) = .25f * (2.0f * get_clamped(dst1, p.x, p.y) + get_clamped(dst1, p.x, p.y - 1) + get_clamped(dst1, p.x, p.y + 1));
	return dst2;
}

template<class T>
T& get_wrapZeros(Array2D<T>& src, int x, int y)
{
	if (x < 0 || y < 0 || x >= src.w || y >= src.h)
	{
		return ::zero<T>();
	}
	return src(x, y);
}
template<class T>
T const& get_wrapZeros(Array2D<T> const& src, int x, int y)
{
	if (x < 0 || y < 0 || x >= src.w || y >= src.h)
	{
		return zero<T>();
	}
	return src(x, y);
}
template<class T, class FetchFunc = WrapModes::DefaultImpl>
vec2 gradient_i(Array2D<T>& src, ivec2 const& p)
{
	//if(p.x<1||p.y<1||p.x>=src.w-1||p.y>=src.h-1)
	//	return vec2::zero();
	vec2 gradient;
	gradient.x = (FetchFunc::fetch(src, p.x + 1, p.y) - FetchFunc::fetch(src, p.x - 1, p.y)) / 2.0f;
	gradient.y = (FetchFunc::fetch(src, p.x, p.y + 1) - FetchFunc::fetch(src, p.x, p.y - 1)) / 2.0f;
	return gradient;
}
template<class T, class FetchFunc>
vec2 gradient_i_nodiv(Array2D<T>& src, ivec2 const& p)
{
	vec2 gradient(
		FetchFunc::fetch(src, p.x + 1, p.y) - FetchFunc::fetch(src, p.x - 1, p.y),
		FetchFunc::fetch(src, p.x, p.y + 1) - FetchFunc::fetch(src, p.x, p.y - 1));
	return gradient;
}
template<class T, class FetchFunc>
Array2D<vec2> get_gradients(Array2D<T>& src)
{
	auto src2 = src.clone();
	forxy(src2)
		src2(p) /= 2.0f;
	Array2D<vec2> gradients(src.w, src.h);
	for (int x = 0; x < src.w; x++)
	{
		gradients(x, 0) = gradient_i_nodiv<T, FetchFunc>(src2, ivec2(x, 0));
		gradients(x, src.h - 1) = gradient_i_nodiv<T, FetchFunc>(src2, ivec2(x, src.h - 1));
	}
	for (int y = 1; y < src.h - 1; y++)
	{
		gradients(0, y) = gradient_i_nodiv<T, FetchFunc>(src2, ivec2(0, y));
		gradients(src.w - 1, y) = gradient_i_nodiv<T, FetchFunc>(src2, ivec2(src.w - 1, y));
	}
	for (int y = 1; y < src.h - 1; y++) {
		for (int x = 1; x < src.w - 1; x++) {
			gradients(x, y) = gradient_i_nodiv<T, WrapModes::NoWrap>(src2, ivec2(x, y));
		}
	}
	return gradients;
}
template<class T>
Array2D<vec2> get_gradients(Array2D<T> src)
{
	return get_gradients<T, WrapModes::DefaultImpl>(src);
}

void mm(string desc, Array2D<float> arr);
void mm(string desc, Array2D<vec3> arr);
void mm(string desc, Array2D<vec2> arr);
//get_wrapZeros

vector<float> getGaussianKernel(int ksize, float sigma); // ksize must be odd

float sigmaFromKsize(float ksize);

float ksizeFromSigma(float sigma);

// KS means it has ksize and sigma args
template<class T, class FetchFunc>
Array2D<T> separableConvolve(Array2D<T> src, vector<float>& kernel) {
	int ksize = kernel.size();
	int r = ksize / 2;

	T zero = ::zero<T>();
	Array2D<T> dst1(src.w, src.h);
	Array2D<T> dst2(src.w, src.h);

	int w = src.w, h = src.h;

	// vertical

	for (int y = 0; y < h; y++)
	{
		auto blurVert = [&](int x0, int x1) {
			// guard against w<r
			x0 = std::max(x0, 0);
			x1 = std::min(x1, w);

			for (int x = x0; x < x1; x++)
			{
				T sum = zero;
				for (int xadd = -r; xadd <= r; xadd++)
				{
					sum += kernel[xadd + r] * (FetchFunc::fetch<T>(src, x + xadd, y));
				}
				dst1(x, y) = sum;
			}
		};


		blurVert(0, r);
		blurVert(w - r, w);
		for (int x = r; x < w - r; x++)
		{
			T sum = zero;
			for (int xadd = -r; xadd <= r; xadd++)
			{
				sum += kernel[xadd + r] * src(x + xadd, y);
			}
			dst1(x, y) = sum;
		}
	}

	// horizontal
	for (int x = 0; x < w; x++)
	{
		auto blurHorz = [&](int y0, int y1) {
			// guard against h<r
			y0 = std::max(y0, 0);
			y1 = std::min(y1, h);
			for (int y = y0; y < y1; y++)
			{
				T sum = zero;
				for (int yadd = -r; yadd <= r; yadd++)
				{
					sum += kernel[yadd + r] * FetchFunc::fetch<T>(dst1, x, y + yadd);
				}
				dst2(x, y) = sum;
			}
		};

		blurHorz(0, r);
		blurHorz(h - r, h);
		for (int y = r; y < h - r; y++)
		{
			T sum = zero;
			for (int yadd = -r; yadd <= r; yadd++)
			{
				sum += kernel[yadd + r] * dst1(x, y + yadd);
			}
			dst2(x, y) = sum;
		}
	}
	return dst2;
}
// one-arg version for backward compatibility
template<class T, class FetchFunc>
Array2D<T> gaussianBlur(Array2D<T> src, int ksize) { // ksize must be odd.
	auto kernel = getGaussianKernel(ksize, sigmaFromKsize(ksize));
	return separableConvolve<T, FetchFunc>(src, kernel);
}
template<class T>
Array2D<T> gaussianBlur(Array2D<T> src, int ksize) {
	return gaussianBlur<T, WrapModes::DefaultImpl>(src, ksize);
}


Array2D<vec3> merge(vector<Array2D<float> > channels);

Array2D<float> div(Array2D<vec2> a);


Array2D<vec2> gradientForward(Array2D<float> a);

Array2D<float> divBackward(Array2D<vec2> a);