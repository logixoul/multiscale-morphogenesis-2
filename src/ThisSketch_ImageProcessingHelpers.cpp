#include "precompiled.h"
#include "ThisSketch_ImageProcessingHelpers.h"
#include "stuff.h"

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
		state = ::resize(state, newSize, filter);
	}
	return scales;
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
