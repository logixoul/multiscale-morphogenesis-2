#pragma once
#include "precompiled.h"
#include "util.h"

struct PerlinCalculator2D
{
	vector<vec3> gradients;
	static const int size = 16;
	PerlinCalculator2D();
	vec3& getGradient(ivec3 const& v) {
		return gradients[v.z * size * size + v.y * size + v.x];
	}
	// p loops in [0, 1]
	float fade(float f) { return smoothstep(0.0, 1.0, f); }
	float pos_mod(float a, float b) { float c = fmod(a, b); if(c < 0.0f) c += b; return c; }
	float calcAt(vec3 const& p);
};
