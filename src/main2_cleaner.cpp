#include "precompiled.h"
//#include "ciextra.h"
#include "util.h"
//#include "shade.h"
#include "stuff.h"
#include "Array2D_imageProc.h"
#include "gpgpu.h"
#include "cfg2.h"
#include "sw.h"

#include "stefanfw.h"

#include "CrossThreadCallQueue.h"

static vec2 perpLeft(vec2 const& v) {
	return vec2(-v.y, v.x);
};

using Img = Array2D<float>;
static Img subtract(Img const& a, Img const& b) {
	Img result = a.clone();
	for (int i = 0; i < result.area; i++) {
		result.data[i] -= b.data[i];
	}
	return result;
}
static Img add(Img const& a, Img const& b) {
	Img result = a.clone();
	for (int i = 0; i < result.area; i++) {
		result.data[i] += b.data[i];
	}
	return result;
}
static Img multiply(Img const& a, Img const& b) {
	Img result = a.clone();
	for (int i = 0; i < result.area; i++) {
		result.data[i] *= b.data[i];
	}
	return result;
}
static Img multiply(Img const& a, float scalar) {
	Img result = a.clone();
	for (int i = 0; i < result.area; i++) {
		result.data[i] *= scalar;
	}
	return result;
}
template<class T, class FetchFunc>
static Array2D<T> gaussianBlur3x3(Array2D<T> src) {
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
	// Sobel uses a [1,2,1] weighting perpendicular to the derivative direction,
	// giving better angular resolution than simple finite differences.
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

// implemented by AI
template<class T, class FetchFunc>
static T getBicubic(Array2D<T>& src, vec2 p) {
	vec2 f = glm::floor(p);
	vec2 t = p - f;
	int ix = (int)f.x;
	int iy = (int)f.y;

	auto w = [](float t) -> vec4 {
		// Catmull-Rom weights
		float t2 = t * t;
		float t3 = t2 * t;
		return vec4(
			-0.5f * t3 + t2 - 0.5f * t,
			1.5f * t3 - 2.5f * t2 + 1.0f,
			-1.5f * t3 + 2.0f * t2 + 0.5f * t,
			0.5f * t3 - 0.5f * t2
		);
		};

	vec4 wx = w(t.x);
	vec4 wy = w(t.y);

	T result = T(0);
	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			result += wx[i] * wy[j] * FetchFunc::fetch(src, ix - 1 + i, iy - 1 + j);
		}
	}
	return result;
}

#if 0
static Array2D<vec3> resize(Array2D<vec3> src, ivec2 dstSize, const ci::FilterBase& filter)
{
	ci::SurfaceT<float> tmpSurface(
		(float*)src.data, src.w, src.h, /*rowBytes*/sizeof(vec3) * src.w, ci::SurfaceChannelOrder::RGB);
	auto resizedSurface = ci::ip::resizeCopy(tmpSurface, tmpSurface.getBounds(), dstSize, filter);
	Array2D<vec3> resultArray = resizedSurface;
	return resultArray;
}
#endif

static Array2D<float> resize(Array2D<float> src, ivec2 dstSize, const ci::FilterBase& filter)
{
	ci::ChannelT<float> tmpSurface(
		src.w, src.h, /*rowBytes*/sizeof(float) * src.w, 1, src.data);
	ci::ChannelT<float> resizedSurface(dstSize.x, dstSize.y);
	ci::ip::resize(tmpSurface, &resizedSurface, filter);
	Array2D<float> resultArray = resizedSurface;
	return resultArray;
}

static std::vector<Img> buildGaussianPyramid(Img src) {
	std::vector<Img> scales;
	int size = std::min(src.w, src.h);
	auto state = src.clone();
	static const auto filter = ci::FilterGaussian();
	while (size > 2)
	{
		scales.push_back(state);
		state = ::resize(state, state.Size() / 2, filter);
		size /= 2;
	}
	return scales;
}

static gl::TextureRef redToLuminance(gl::TextureRef const& in) {
	return shade2(in,
		"_out.rgb = vec3(fetch1());",
		ShadeOpts().ifmt(GL_RGBA16F)
	);
}
static float blendHardLight(float base, float blend) {
	if (blend < 0.5f) {
		// Multiply: darkens the image based on the blend layer
		return 2.0f * base * blend;
	}
	else {
		// Screen: lightens the image based on the blend layer
		return 1.0f - 2.0f * (1.0f - base) * (1.0f - blend);
	}
}

int wsx = 700, wsy = 700;
int sx = 256;
int sy = 256;
Array2D<float> img(sx, sy);
bool pause2 = false;

struct SApp : App {
	struct Options {
		float morphogenesisStrength;
		float contrastizeStrength;
		float blendWeaken;
		float weightFactor;
		bool multiscale;
		bool binarizePostprocessing;
		float highPassStrength;

		static Options get() {
			return Options{
				cfg2::getFloat("morphogenesis", .02, 0.068, 20, 0.658, ImGuiSliderFlags_Logarithmic),
				cfg2::getFloat("contrastizeFactor", 0.01f, 1.0, 10, 1.0f),
				cfg2::getFloat("blendWeaken", 0.01f, 0.1, .5f, .490f),
				cfg2::getFloat("weightFactor", 0.1f, 0.01f, 60.0f, 0.1f, ImGuiSliderFlags_Logarithmic),
				cfg2::getBool("multiscale", true),
				cfg2::getBool("binarizePostprocessing", true),
				cfg2::getFloat("highPassStrength", 0.01f, 0.0f, 1.0f, 0.90f)
			};
		}
	};

	void setup()
	{
		reset();
		enableDenormalFlushToZero();
		setWindowSize(wsx, wsy);

		disableGLReadClamp();
		stefanfw::eventHandler.subscribeToEvents(*this);

		cfg2::init();
	}

	void update()
	{
		cfg2::begin();
		stefanfw::beginFrame();
		stefanUpdate();
		stefanDraw();
		stefanfw::endFrame();
		cfg2::end();
	}
	void keyDown(KeyEvent e)
	{
		if (keys['p'] || keys['2'])
		{
			pause2 = !pause2;
		}
		if (keys['r'])
		{
			reset();
		}
		if (e.getChar() == 'd')
		{
			//cfg2::params->isVisible() ? cfg2::params->hide() : cfg2::params->show();
		}
	}
	void reset() {
		forxy(img) {
			img(p) = ::randFloat();
		}
	}
	template<class T, class FetchFunc>
	static float hessianDirectionalSecondDeriv(Array2D<T>& src, ivec2 const& p, vec2 const& d) {
		float fxx = FetchFunc::fetch(src, p.x + 1, p.y) - 2.0f * FetchFunc::fetch(src, p.x, p.y) + FetchFunc::fetch(src, p.x - 1, p.y);
		float fyy = FetchFunc::fetch(src, p.x, p.y + 1) - 2.0f * FetchFunc::fetch(src, p.x, p.y) + FetchFunc::fetch(src, p.x, p.y - 1);
		float fxy = 0.25f * (
			FetchFunc::fetch(src, p.x + 1, p.y + 1)
			- FetchFunc::fetch(src, p.x - 1, p.y + 1)
			- FetchFunc::fetch(src, p.x + 1, p.y - 1)
			+ FetchFunc::fetch(src, p.x - 1, p.y - 1));
		return d.x * d.x * fxx + 2.0f * d.x * d.y * fxy + d.y * d.y * fyy;
	}
	Array2D<float> updateSingleScale(Array2D<float> aImg)
	{
		auto img = aImg.clone();

		//auto blurredImg = gaussianBlur3x3<float, WrapModes::GetClamped>(img);
		auto kernel = getGaussianKernel(3, sigmaFromKsize(3));
		//auto blurredImg = ::separableConvolve<float, WrapModes::GetClamped>(img, kernel);
		auto gradients = ::get_gradients_sobel<float, WrapModes::GetClamped>(img);
		auto img2 = img.clone();
		forxy(img) {
			vec2 const& pf = vec2(p);
			vec2 const& grad = gradients(p);
			vec2 const& gradN = safeNormalized(grad);

			vec2 const& gradNPerp = perpLeft(gradN);

			//float val = img(p);
			//float valLeft = getBicubic<float, WrapModes::GetClamped>(img, pf + gradNPerp);
			//float valRight = getBicubic<float, WrapModes::GetClamped>(img, pf - gradNPerp);
			//float add = (val - (valLeft + valRight) * .5f);
			float add = -hessianDirectionalSecondDeriv<float, WrapModes::GetClamped>(img, p, gradNPerp);
			aaPoint<float, WrapModes::GetClamped>(img2, pf - gradN * add, add * options.morphogenesisStrength);
		}
		//img = multiply(add(img2, blurredImg2), .5);
		//auto blurredImg2 = ::separableConvolve<float, WrapModes::GetClamped>(img2, kernel);
		auto blurredImg2 = ::gaussianBlur3x3<float, WrapModes::GetClamped>(img2);
		img = blurredImg2;
		img = applyVerticalGradient(img);

		return img;
	}
	Img applyVerticalGradient(Img const& img) {
		Img result = ::zeros_like(img);
		forxy(result) {
			float floatY = p.y / (float)result.h;
			floatY = glm::mix(options.blendWeaken, 1.0f - options.blendWeaken, floatY);
			result(p) = blendHardLight(img(p), floatY);
		}
		return result;
	}
	float getLevelWeight(int level, int maxLevel) const {
		float iNormalized = level / float(maxLevel - 1);
		return exp(options.weightFactor*iNormalized);
	}
	std::vector<float> getLevelWeights(int numLevels) const {
		std::vector<float> result;
		for (int i = 0; i < numLevels; i++) {
			result.push_back(getLevelWeight(i, numLevels));
		}
		float sum = std::accumulate(result.begin(), result.end(), 0.0f);
		for(auto& weight : result) {
		//	weight /= sum;
		}
		return result;
	}
	Img multiscaleApply(Img src, function<Img(Img)> func) {
		std::vector<Img> origScales = ::buildGaussianPyramid(src);
		std::vector<Img> updatedScales(origScales.size());
		static const auto filter = ci::FilterGaussian();
		/*for (auto s : origScales)
		{
			updatedScales.push_back(func(s));
		}*/
		const int last = origScales.size() - 1;
		updatedScales[last] = func(origScales[last]);
		auto weights = getLevelWeights(origScales.size());
		for (int i = updatedScales.size() - 1; i >= 1; i--) {
			auto diff = ::subtract(updatedScales[i], origScales[i]);
			diff = ::multiply(diff, weights[i]);
			auto upscaledDiff = ::resize(diff, origScales[i - 1].Size(), filter);
			auto& nextScale = updatedScales[i - 1];
			nextScale = ::add(origScales[i - 1], upscaledDiff);
			nextScale = func(nextScale);
			forxy(nextScale) {
				//nextScale(p) = mulContrastize(nextScale(p), options.contrastizeStrength);
			}
		}
		return updatedScales[0];
	}
	Options options;
	void stefanUpdate() {
		this->options = Options::get();

		if (pause2) {
			return;
		}
		if(options.multiscale)
			img = multiscaleApply(img, [this](auto arg) { return updateSingleScale(arg); });
		else
			img = updateSingleScale(img);

		//img = to01(img);
		forxy(img) {
			//img(p) = mulContrastize(img(p), options.contrastizeStrength);
		}
	}

	static float mulContrastize(float i, float contrastizeStrength) {
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
	gl::TextureRef postprocess() {
		auto img2 = img.clone();
		forxy(img2) img2(p) -= .5f;
		auto accum = zeros_like(img);
		float sumw = 0.0f;
		for (int kernelRadius = 3; kernelRadius <= 15; kernelRadius *= 2) {
			auto imgb = gaussianBlur<float, WrapModes::GetClamped>(img2, kernelRadius * 2 + 1);
			accum = add(accum, imgb);
			sumw++;
		}
		accum = multiply(accum, 1.0f / sumw);
		auto texb = gtex(accum);
		auto tex = gtex(img2);
		tex = shade2(tex, texb,
			"float f = fetch1();"
			"float fBlurred = fetch1(tex2);"
			"float highPassed = f - fBlurred*highPassStrength;"
			"float fw = fwidth(highPassed);"
			"highPassed = smoothstep(-fw/2.0, fw/2.0, highPassed);"
			"_out.r = mix(f + .5, highPassed, .5);",
			ShadeOpts().dstRectSize(getWindowSize()).uniform("highPassStrength", options.highPassStrength)
		);
		return tex;
	}
	gl::TextureRef postprocessV2() {
		auto imgClamped = img.clone();
		forxy(imgClamped) imgClamped(p) = ci::constrain(imgClamped(p), 0.0f, 1.0f);
		auto pyramid = buildGaussianPyramid(imgClamped);
		auto stateTex = maketex(img.w, img.h, GL_R16F, false, true);
		for(int i = pyramid.size() - 1; i >= 0; i--) {
			auto& thisLevel = pyramid[i];
			auto thisLevelTex = gtex(thisLevel);
			thisLevelTex = shade2(thisLevelTex,
				"float f = fetch1();"
				"float fw = fwidth(f);"
				"f = smoothstep(.5-fw/2.0, .5+fw/2.0, f);"
				"_out.r = f;", ShadeOpts().dstRectSize(img.Size()));
			stateTex = op(stateTex) + thisLevelTex;
		}
		stateTex = op(stateTex) / float(pyramid.size());
		return stateTex;
	}
	void stefanDraw()
	{
		gl::setMatricesWindow(vec2(wsx, wsy), false);
		gl::clear(ColorA::black(), true);
		gl::disableDepthRead();

		sw::timeit("draw", [&]() {
			gl::TextureRef tex = gtex(img);
			if (options.binarizePostprocessing) {
				tex = postprocessV2();
			}
			gl::draw(redToLuminance(tex), getWindowBounds());
		});
	}
};
CrossThreadCallQueue* gMainThreadCallQueue;

CINDER_APP(SApp, RendererGl(),
	[&](ci::app::App::Settings* settings)
	{
		settings->setConsoleWindowEnabled(true);
	})