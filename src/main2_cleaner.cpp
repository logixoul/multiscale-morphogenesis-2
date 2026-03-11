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

template<class T>
struct compareVec
{
	bool operator() (const glm::vec<2, T>& lhs, const glm::vec<2, T>& rhs) const
	{
		if (lhs.x < rhs.x) return -1;
		if (lhs.x > rhs.x) return 1;
		return lhs.y < rhs.y;
	}
};

Array2D<vec3> resize(Array2D<vec3> src, ivec2 dstSize, const ci::FilterBase& filter)
{
	ci::SurfaceT<float> tmpSurface(
		(float*)src.data, src.w, src.h, /*rowBytes*/sizeof(vec3) * src.w, ci::SurfaceChannelOrder::RGB);
	auto resizedSurface = ci::ip::resizeCopy(tmpSurface, tmpSurface.getBounds(), dstSize, filter);
	Array2D<vec3> resultArray = resizedSurface;
	return resultArray;
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
gl::TextureRef redToLuminance(gl::TextureRef const& in) {
	return shade2(in,
		"_out.rgb = vec3(fetch1());",
		ShadeOpts().ifmt(GL_RGBA16F)
	);
}

int wsx = 700, wsy = 700;
int sx = 256;
int sy = 256;
Array2D<float> img(sx, sy);
bool pause2 = false;
std::map<int, gl::TextureRef> texs;

struct SApp : App {
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

		auto gradients = ::get_gradients<float, WrapModes::GetClamped>(img);
		static const auto perpLeft = [&](vec2 v) { return vec2(-v.y, v.x); };
		auto img2 = img.clone();
		for (int x = 0; x < img.w; x++)
		{
			for (int y = 0; y < img.h; y++)
			{
				vec2 p = vec2(x, y);
				vec2 grad = safeNormalized(gradients(x, y));

				vec2 gradP = perpLeft(grad);

				float val = img(x, y);
				float valLeft = getBilinear<float, WrapModes::GetClamped>(img, p + gradP);
				float valRight = getBilinear<float, WrapModes::GetClamped>(img, p - gradP);
				float add = (val - (valLeft + valRight) * .5f);
				aaPoint<float, WrapModes::GetClamped>(img2, p - grad * add, add * abc);
			}
		}
		img = gauss3Better<float, WrapModes::GetClamped>(img2);
		
		forxy(img) {
			float floatY = p.y / (float)img.h;
			floatY = glm::mix(blendWeaken, 1.0f - blendWeaken, floatY);
			if (floatY < .5) {
				img(p) *= floatY * 2;
			}
			else {
				img(p) = glm::mix(img(p), 1.0f, (floatY - 0.5f) * 2);
			}
		}
		return img;
	}
	template<class T, class FetchFunc>
	static Array2D<T> gauss3Better(Array2D<T> src) {
		T zero = ::zero<T>();
		Array2D<T> dst1(src.w, src.h);
		Array2D<T> dst2(src.w, src.h);
		forxy(dst1)
			dst1(p) = .25f * (2 * FetchFunc::fetch(src, p.x, p.y) + FetchFunc::fetch(src, p.x - 1, p.y) + FetchFunc::fetch(src, p.x + 1, p.y));
		forxy(dst2)
			dst2(p) = .25f * (2 * FetchFunc::fetch(dst1, p.x, p.y) + FetchFunc::fetch(dst1, p.x, p.y - 1) + FetchFunc::fetch(dst1, p.x, p.y + 1));
		return dst2;
	}
	using Img = Array2D<float>;
	Img multiscaleApply(Img src, function<Img(Img)> func) {
		int size = std::min(src.w, src.h);
		auto state = src.clone();
		vector<Img> scales;
		auto filter = ci::FilterGaussian();
		
		while (size > 2)
		{
			scales.push_back(state);
			state = ::resize(state, state.Size() / 2, filter);
			size /= 2;
		}
		vector<Img> origScales = scales;
		for (auto& s : origScales) s = s.clone();
		int lastLevel = 0;
		for (int i = scales.size() - 1; i >= lastLevel; i--) {
			auto& thisScale = scales[i];
			auto& thisOrigScale = origScales[i];
			auto transformed = func(thisScale);
			auto diff = empty_like(transformed);
			for (int j = 0; j < diff.area; j++) {
				diff.data[j] = transformed.data[j] - thisOrigScale.data[j];
			}
			float iNormalized = -1+2*i / float(scales.size() - 1);
			float w = exp(weightFactor*iNormalized);
			forxy(diff) {
				diff(p) *= w;
			}
			if (i == lastLevel)
			{
				for (int j = 0; j < transformed.area; j++) {
					scales[lastLevel].data[j] = thisOrigScale.data[j] + diff.data[j];//.clone();
				}
				break;
			}
			auto& nextScaleUp = scales[i - 1];
			auto upscaledDiff = ::resize(diff, nextScaleUp.Size(), filter);
			forxy(nextScaleUp) {
				nextScaleUp(p) += upscaledDiff(p);
			}
		}
		return scales[lastLevel];
	}
	float abc;
	float contrastizeFactor;
	float blendWeaken;
	float weightFactor;
	
	void stefanUpdate() {
		abc = cfg2::getFloat("morphogenesis", .02, 0.068, 20, 1.35, ImGuiSliderFlags_Logarithmic);
		contrastizeFactor = cfg2::getFloat("contrastizeFactor", 0.01f, 1.0, 10, 1.0f);
		blendWeaken = cfg2::getFloat("blendWeaken", 0.01f, 0.1, .5f, .45f);
		weightFactor = cfg2::getFloat("weightFactor", 0.1f, 0.1, 60.0f, 0.37, ImGuiSliderFlags_Logarithmic);
		bool multiscale = cfg2::getBool("multiscale", true);
		
		if (pause2) {
			return;
		}
		if(multiscale)
			img = multiscaleApply(img, [this](auto arg) { return updateSingleScale(arg); });
		else
			img = updateSingleScale(img);

		img = to01(img);
		forxy(img) {
			auto& c = img(p);
			c = ci::constrain(c, 0.0f, 1.0f);
			c = mulContrastize(c, contrastizeFactor);
		}

	}

	static float mulContrastize(float i, float contrastizeFactor) {
		const bool invert = i > .5f;
		if (invert) {
			i = 1.0f - i;
		}
		i *= 2.0f;
		i = pow(i, contrastizeFactor);
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