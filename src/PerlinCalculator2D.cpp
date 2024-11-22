#include "precompiled.h"
#include "PerlinCalculator2D.h"

float PerlinCalculator2D::calcAt(vec3 const& p){
	vec3 pMod = vec3(pos_mod(p.x, 1.0f), pos_mod(p.y, 1.0f), pos_mod(p.z, 1.0f));
	vec3 pScaled = pMod * float(size - 1);
	vec3 floor(std::floor(pScaled.x), std::floor(pScaled.y), std::floor(pScaled.z));
	ivec3 ifloor = floor;

	vec3 fract = pScaled - floor;
	float fx = fade(fract.x);
	float fy = fade(fract.y);
	float fz = fade(fract.z);
	float dot[2][2][2];
	for(int x = 0; x <= 1; x++)
		for(int y = 0; y <= 1; y++)
			for(int z = 0; z <= 1; z++)
			{
				vec3 const& grad = getGradient(ifloor + ivec3(x, y, z));
				dot[x][y][z] = ci::dot(grad, vec3(x, y, z) - fract);
			}
		
	// back->front = y, bottom->top = z, left->right = x
	float lerpXTopBack = lerp(dot[0][0][1], dot[1][0][1], fx);
	float lerpXTopFront = lerp(dot[0][1][1], dot[1][1][1], fx);
	float lerpXBottomBack = lerp(dot[0][0][0], dot[1][0][0], fx);
	float lerpXBottomFront = lerp(dot[0][1][0], dot[1][1][0], fx);
	float lerpZBack = lerp(lerpXBottomBack, lerpXTopBack, fz);
	float lerpZFront = lerp(lerpXBottomFront, lerpXTopFront, fz);
	float lerpY = lerp(lerpZBack, lerpZFront, fy);
	return lerpY;
}

PerlinCalculator2D::PerlinCalculator2D()
{
	gradients = vector<vec3>(size*size*size);

	for(int x = 0; x < size; x++)
	{
		for(int y = 0; y < size; y++)
		{
			for(int z = 0; z < size; z++)
			{
				auto& gradient = getGradient(ivec3(x, y, z));
				gradient = ci::Rand::randVec3();
			}
		}
	}
	for(int x = 0; x < size; x++)
	{
		for(int y = 0; y < size; y++)
		{
			for(int z = 0; z < size; z++)
			{
				auto& gradient = getGradient(ivec3(x, y, z));
				int wx = x % (size - 1);
				int wy = y % (size - 1);
				int wz = z % (size - 1);
				gradient = getGradient(ivec3(wx, wy, wz));
			}
		}
	}
}