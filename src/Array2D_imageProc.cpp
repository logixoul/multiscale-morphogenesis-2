#include "precompiled.h"
#include "Array2D_imageProc.h"

#if 0
Array2D<vec3> resize(Array2D<vec3> src, ivec2 dstSize, const ci::FilterBase &filter)
{
	ci::SurfaceT<float> tmpSurface(
		(float*)src.data, src.w, src.h, /*rowBytes*/sizeof(vec3) * src.w, ci::SurfaceChannelOrder::RGB);
	auto resizedSurface = ci::ip::resizeCopy(tmpSurface, tmpSurface.getBounds(), dstSize, filter);
	Array2D<vec3> resultArray = resizedSurface;
	return resultArray;
}

Array2D<float> resize(Array2D<float> src, ivec2 dstSize, const ci::FilterBase &filter)
{
	ci::ChannelT<float> tmpSurface(
		src.w, src.h, /*rowBytes*/sizeof(float) * src.w, 1, src.data);
	ci::ChannelT<float> resizedSurface(dstSize.x, dstSize.y);
	ci::ip::resize(tmpSurface, &resizedSurface, filter);
	Array2D<float> resultArray = resizedSurface;
	return resultArray;
}
#endif

// todo: consolidate the following mm funcs
void mm(string desc, Array2D<float> arr) {
	if (desc != "") {
		cout << "[" << desc << "] ";
	}
	cout << "min: " << *std::min_element(arr.begin(), arr.end()) << ", "
		<< "max: " << *std::max_element(arr.begin(), arr.end()) << endl;
}
void mm(string desc, Array2D<vec3> arr) {
	if (desc != "") {
		cout << "[" << desc << "] ";
	}
	auto data = (float*)arr.data;
	cout << "min: " << *std::min_element(data, data + arr.area + 2) << ", "
		<< "max: " << *std::max_element(data, data + arr.area + 2) << endl;
}
void mm(string desc, Array2D<vec2> arr) {
	if (desc != "") {
		cout << "[" << desc << "] ";
	}
	auto data = (float*)arr.data;
	cout << "min: " << *std::min_element(data, data + arr.area + 1) << ", "
		<< "max: " << *std::max_element(data, data + arr.area + 1) << endl;
}

float sq(float f) {
	return f * f;
}

vector<float> getGaussianKernel(int ksize, float sigma) {
	vector<float> result;
	int r = ksize / 2;
	float sum = 0.0f;
	for (int i = -r; i <= r; i++) {
		float exponent = -(i*i / sq(2 * sigma));
		float val = exp(exponent);
		sum += val;
		result.push_back(val);
	}
	for (int i = 0; i < result.size(); i++) {
		result[i] /= sum;
	}
	return result;
}

float sigmaFromKsize(float ksize) {
	float sigma = 0.3*((ksize - 1)*0.5 - 1) + 0.8;
	return sigma;
}

float ksizeFromSigma(float sigma) {
	// ceil just to be sure
	int ksize = ceil(((sigma - 0.8) / 0.3 + 1) / 0.5 + 1);
	if (ksize % 2 == 0)
		ksize++;
	return ksize;
}

Array2D<vec2> gradientForward(Array2D<float> a) {
	throw 0;
	/*return ::map(a, [&](ivec2 p) -> vec2 {
		return vec2(
			(a.wr(p.x + 1, p.y) - a.wr(p.x, p.y)) / 1.0f,
			(a.wr(p.x, p.y + 1) - a.wr(p.x, p.y)) / 1.0f
		);
	});*/
}

Array2D<float> divBackward(Array2D<vec2> a) {
	throw 0;
	/*return ::map(a, [&](ivec2 p) -> float {
		auto dGx_dx = (a.wr(p.x, p.y).x - a.wr(p.x - 1, p.y).x);
		auto dGy_dy = (a.wr(p.x, p.y).y - a.wr(p.x, p.y - 1).y);
		return dGx_dx + dGy_dy;
	})*/;
}

vector<Array2D<float> > split(Array2D<vec3> arr) {
	Array2D<float> r(arr.w, arr.h);
	Array2D<float> g(arr.w, arr.h);
	Array2D<float> b(arr.w, arr.h);
	forxy(arr) {
		r(p) = arr(p).x;
		g(p) = arr(p).y;
		b(p) = arr(p).z;
	}
	vector<Array2D<float> > result;
	result.push_back(r);
	result.push_back(g);
	result.push_back(b);
	return result;
}


Array2D<vec3> merge(vector<Array2D<float> > channels) {
	Array2D<float>& r = channels[0];
	Array2D<float>& g = channels[1];
	Array2D<float>& b = channels[2];
	Array2D<vec3> result(r.w, r.h);
	forxy(result) {
		result(p) = vec3(r(p), g(p), b(p));
	}
	return result;
}

Array2D<float> div(Array2D<vec2> a) {
	throw 0;
	/*return ::map(a, [&](ivec2 p) -> float {
		auto dGx_dx = (a.wr(p.x + 1, p.y).x - a.wr(p.x - 1, p.y).x) / 2.0f;
		auto dGy_dy = (a.wr(p.x, p.y + 1).y - a.wr(p.x, p.y - 1).y) / 2.0f;
		return dGx_dx + dGy_dy;
	});*/
}

Array2D<float> to01(Array2D<float> a) {
	auto minn = *std::min_element(a.begin(), a.end());
	auto maxx = *std::max_element(a.begin(), a.end());
	auto b = a.clone();
	forxy(b) {
		b(p) -= minn;
		b(p) /= (maxx - minn);
	}
	return b;
}

Array2D<vec3> to01(Array2D<vec3> a, float min, float max) {
	auto b = a.clone();
	forxy(b) {
		b(p) -= vec3(min);
		b(p) /= vec3(max - min);
	}
	return b;
}
