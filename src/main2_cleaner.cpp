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

		static Options get() {
			return Options{
				cfg2::getFloat("morphogenesis", .02, 0.068, 20, 1.35, ImGuiSliderFlags_Logarithmic),
				cfg2::getFloat("contrastizeFactor", 0.01f, 1.0, 10, 1.0f),
				cfg2::getFloat("blendWeaken", 0.01f, 0.1, .5f, .49f),
				cfg2::getFloat("weightFactor", 0.1f, 0.01f, 60.0f, 0.1f, ImGuiSliderFlags_Logarithmic),
				cfg2::getBool("multiscale", true)
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

	Array2D<float> updateSingleScale(Array2D<float> aImg)
	{
		auto img = aImg.clone();

		auto blurredImg = gaussianBlur3x3<float, WrapModes::GetClamped>(img);
		auto gradients = ::get_gradients<float, WrapModes::GetClamped>(blurredImg);
		auto img2 = img.clone();
		forxy(img) {
			vec2 const& pf = vec2(p);
			vec2 const& grad = gradients(p);
			vec2 const& gradN = safeNormalized(gradients(p));

			vec2 const& gradNPerp = perpLeft(gradN);

			float val = img(p);
			float valLeft = getBilinear<float, WrapModes::GetClamped>(blurredImg, pf + gradNPerp);
			float valRight = getBilinear<float, WrapModes::GetClamped>(blurredImg, pf - gradNPerp);
			float add = (val - (valLeft + valRight) * .5f);
			aaPoint<float, WrapModes::GetClamped>(img2, pf - gradN * add, add * options.morphogenesisStrength);
		}
		img = gaussianBlur3x3<float, WrapModes::GetClamped>(img2);
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
		std::vector<Img> updatedScales;
		static const auto filter = ci::FilterGaussian();
		for (auto s : origScales)
		{
			updatedScales.push_back(func(s));
		}
		auto& weights = getLevelWeights(updatedScales.size());
		for (int i = updatedScales.size() - 1; i >= 1; i--) {
			auto diff = ::subtract(updatedScales[i], origScales[i]);
			diff = ::multiply(diff, weights[i]);
			auto upscaledDiff = ::resize(diff, updatedScales[i - 1].Size(), filter);
			updatedScales[i - 1] = ::add(updatedScales[i - 1], upscaledDiff);
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

		img = to01(img);
		forxy(img) {
			auto& c = img(p);
			c = ci::constrain(c, 0.0f, 1.0f);
			c = mulContrastize(c, options.contrastizeStrength);
		}

	}

	static float mulContrastize(float i, float contrastizeStrength) {
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
	void stefanDraw()
	{
		gl::setMatricesWindow(vec2(wsx, wsy), false);
		gl::clear(ColorA::black(), true);
		gl::disableDepthRead();

		sw::timeit("draw", [&]() {
			auto tex = gtex(img);
			/*tex= shade2(tex, "float f = fetch1();"
				"float fw = fwidth(f);"
				"f = smoothstep(0.5-fw/2.0, 0.5+fw/2.0, f);"
				"_out.r = f;",
				ShadeOpts().dstRectSize(getWindowSize())
			);*/
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