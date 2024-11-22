#pragma once

#include "precompiled.h"
#include "util.h"

template<class T>
float cubic(T val0, T val1, T t0, T t1, float x)
{
	return
		lerp(val0, val1, -2*x*x*x+3*x*x)
		+ (x*x*x-2*x*x+x)*t0
		+ (x*x*x-x*x)*t1;
}
Vec4f cubicCoefs(float x)
{
	return Vec4f(
		-.5 * (x*x*x-2*x*x+x),
		+(1.5*x*x*x-2.5*x*x+1),
		+(-1.5*x*x*x+2*x*x+.5*x),
		+.5 * (x*x*x-x*x)
		);
}
template<class T>
T getBicubic(Array2D<T> src, vec2 p)
{
	return getBicubic(src, p.x, p.y);
}
float dot(Vec4f a, Vec4f b) {
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}
template<class T>
T getBicubic(Array2D<T> src, float xn, float yn)
{
	float x = xn * (float)(src.w) - .5f;
	float y = yn * (float)(src.h) - .5f;
	
	int ix = x, iy = y;
	float fx = ix, fy = iy;
	if(x < 0.0f && fx != x) { fx--; ix--; }
	if(y < 0.0f && fy != y) { fy--; iy--; }
	float fractx = x - fx;
	float fracty = y - fy;
	
	Vec4f xCoefs = cubicCoefs(fractx);
	T interpX1_prev = dot( 
		Vec4f(src.wr(ix - 1, iy-1), src.wr(ix, iy-1), src.wr(ix + 1, iy-1), src.wr(ix + 2, iy-1)),
		xCoefs);
	T interpX1 = dot(
		Vec4f(src.wr(ix - 1, iy), src.wr(ix, iy), src.wr(ix + 1, iy), src.wr(ix + 2, iy)),
		xCoefs);
	T interpX2 = dot(
		Vec4f(src.wr(ix - 1, iy+1), src.wr(ix, iy+1), src.wr(ix + 1, iy+1), src.wr(ix + 2, iy+1)),
		xCoefs);
	T interpX2_next = dot(
		Vec4f(src.wr(ix - 1, iy+2), src.wr(ix, iy+2), src.wr(ix + 1, iy+2), src.wr(ix + 2, iy+2)),
		xCoefs);
	T interpY = dot(
		Vec4f(interpX1_prev, interpX1, interpX2, interpX2_next),
		cubicCoefs(fracty));
	return interpY;
}
template<class T>
T getBicubic2(Array2D<T> src, vec2 v)
{
	return getBicubic2(src, v.x, v.y);
}
template<class T>
T getBicubic2(Array2D<T> src, float xn, float yn)
{
	float x = xn * (float)(src.w) - .5f;
	float y = yn * (float)(src.h) - .5f;

	int ix = x, iy = y;
	float fx = ix, fy = iy;
	if(x < 0.0f && fx != x) { fx--; ix--; }
	if(y < 0.0f && fy != y) { fy--; iy--; }
	float fractx = x - fx;
	
	Vec4f xCoefs = cubicCoefs(fractx);
	T interpX1 = dot(
		Vec4f(src.wr(ix - 1, iy), src.wr(ix, iy), src.wr(ix + 1, iy), src.wr(ix + 2, iy)),
		xCoefs);
	return interpX1;
}
