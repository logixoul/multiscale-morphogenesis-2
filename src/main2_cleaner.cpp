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

inline Array2D<float> to01_Cut(Array2D<float> in) {
	Array2D<float> tmp = in.clone();
	std::sort(tmp.begin(), tmp.end());
	//float div = constrain<float>(mouseX * 100, 1, 100);
	float div = 100;
	float minn = tmp.data[int(tmp.area / div)];
	float maxx = tmp.data[int(tmp.area - 1 - tmp.area / div)];
	auto result = in.clone();
	forxy(result) {
		result(p) -= minn;
		result(p) /= (maxx - minn);
		result(p) = constrain<float>(result(p), 0, 1);
	}
	return result;
}

gl::TextureRef get_gradients_tex_v2(gl::TextureRef src, GLuint wrapS, GLuint wrapT) {
	GPU_SCOPE("get_gradients_tex");
	glActiveTexture(GL_TEXTURE0);
	::bindTexture(src);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
	return shade2(src,
		"	float srcL=fetch1(tex,tc+tsize*vec2(-1.0,0.0));"
		"	float srcR=fetch1(tex,tc+tsize*vec2(1.0,0.0));"
		"	float srcT=fetch1(tex,tc+tsize*vec2(0.0,-1.0));"
		"	float srcB=fetch1(tex,tc+tsize*vec2(0.0,1.0));"
		"	float dx=(srcR-srcL)/2.0;"
		"	float dy=(srcB-srcT)/2.0;"
		"	_out.xy=vec2(dx,dy);"
		,
		ShadeOpts().ifmt(GL_RG16F)
	);
}

gl::TextureRef get_gradients_tex_v3(gl::TextureRef src, GLuint wrapS, GLuint wrapT) {
	GPU_SCOPE("get_gradients_tex");
	glActiveTexture(GL_TEXTURE0);
	::bindTexture(src);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
	return shade2(src,
		// Sobel 3x3 kernel for X and Y
		"float nw = fetch1(tex, tc + tsize * vec2(-1.0, -1.0));"
		"float n  = fetch1(tex, tc + tsize * vec2(0.0, -1.0));"
		"float ne = fetch1(tex, tc + tsize * vec2(1.0, -1.0));"
		"float w  = fetch1(tex, tc + tsize * vec2(-1.0, 0.0));"
		"float e  = fetch1(tex, tc + tsize * vec2(1.0, 0.0));"
		"float sw = fetch1(tex, tc + tsize * vec2(-1.0, 1.0));"
		"float s  = fetch1(tex, tc + tsize * vec2(0.0, 1.0));"
		"float se = fetch1(tex, tc + tsize * vec2(1.0, 1.0));"
		"float gx = (ne + 2.0 * e + se) - (nw + 2.0 * w + sw);"
		"float gy = (sw + 2.0 * s + se) - (nw + 2.0 * n + ne);"
		// normalize by 8 (sum of absolute kernel weights) to keep scale similar to central differences
		"_out.xy = vec2(gx, gy) / 8.0;",
		ShadeOpts().ifmt(GL_RG16F)
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

	typedef Array2D<float> Img;
	Img update_1_scale(Img aImg)
	{
		auto img = aImg.clone();

		auto tex = gtex(img);
		gl::TextureRef gradientsTex;
		//gradientsTex = get_gradients_tex_v2(tex, GL_REPEAT, GL_CLAMP_TO_EDGE);
		gradientsTex = get_gradients_tex_v3(tex, GL_REPEAT, GL_CLAMP_TO_EDGE);

		static std::map<glm::ivec2, gl::TextureRef, compareVec<int>> changeMap; // velocity of change
		auto accTex = shade2(tex, gradientsTex, // acceleration
			"vec2 grad = fetch2(tex2);"
			"vec2 dir = perpLeft(safeNormalized(grad));"
			""
			"float val = fetch1();"
			"float valLeft = fetch1(tex, tc + tsize * dir);"
			"float valRight = fetch1(tex, tc - tsize * dir);"
			"float add = (val - (valLeft + valRight) * .5f);"
			"_out.r = add * abc;"
			, ShadeOpts().uniform("abc", abc),
			"vec2 perpLeft(vec2 v) {"
			"	return vec2(-v.y, v.x);"
			"}"
		);
		if (changeMap.find(tex->getSize()) == changeMap.end()) {
			changeMap[tex->getSize()] = maketex(tex->getWidth(), tex->getHeight(), GL_R16F, false, true);
		}
		auto changeTex = changeMap[tex->getSize()];
		changeTex = op(changeTex) + accTex;
		tex = op(tex) + changeTex;


		auto texb = tex;
		for (int i = 0; i < 3; i++) {
			texb->setWrap(GL_REPEAT, GL_CLAMP_TO_EDGE);
			texb = gauss3tex(texb);
		}
		//tex = texb;

		tex = shade2(tex, texb,
			"float f = fetch1();"
			"float fb = fetch1(tex2);"
			"_out.r = mix(f, fb, .8f);"
		);
		img = gettexdata<float>(tex, GL_RED, GL_FLOAT);
		img = ::to01(img);

		float sum = ::accumulate(img.begin(), img.end(), 0.0f);
		float avg = sum / (float)img.area;
		forxy(img)
		{
			img(p) += .5f - avg;
		}
		forxy(img) {
			float floatY = p.y / (float)img.h;
			floatY = glm::mix(blendWeaken, 1.0f - blendWeaken, floatY);
			floatY = std::max(0.0f, std::min(1.0f, floatY));
			if (floatY < .5) {
				img(p) *= floatY * 2;
			}
			else {
				img(p) = glm::mix(img(p), 1.0f, (floatY - 0.5f) * 2);
			}
		}
		return img;
	}

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
			//texs[i] = gtex(scales[i]);
			auto& thisScale = scales[i];
			auto& thisOrigScale = origScales[i];
			auto transformed = func(thisScale);
			auto diff = empty_like(transformed);
			sw::timeit("::map", [&]() {
#pragma omp parallel for
				for (int j = 0; j < diff.area; j++) {
					diff.data[j] = transformed.data[j] - thisOrigScale.data[j];
				}
				});
			float w = 1.0f - pow(i / float(scales.size() - 1), weightFactor);
			w = std::max(0.0f, std::min(1.0f, w));
			//float w = exp(-3+6*i / float(scales.size() - 1));
			sw::timeit("2 loops", [&]() {
				forxy(diff) {
					diff(p) *= w;
				}
				});
			if (i == lastLevel)
			{
				sw::timeit("::map", [&]() {
#pragma omp parallel for
					for (int j = 0; j < transformed.area; j++) {
						scales[lastLevel].data[j] = thisOrigScale.data[j] + diff.data[j];//.clone();
					}
					});
				break;
			}
			auto& nextScaleUp = scales[i - 1];
			//texs[i] = gtex(::resize(transformed, nextScaleUp.Size(), filter));
			auto upscaledDiff = ::resize(diff, nextScaleUp.Size(), filter);
			sw::timeit("2 loops", [&]() {
				forxy(nextScaleUp) {
					nextScaleUp(p) += upscaledDiff(p);
				}
				});
		}
		return scales[lastLevel];
	}
	float abc;
	float contrastizeFactor;
	float blendWeaken;
	float weightFactor;
	void stefanUpdate() {
		abc = cfg2::getFloat("morphogenesis", .02, 0.068, 20, 2.418, ImGuiSliderFlags_Logarithmic);
		contrastizeFactor = cfg2::getFloat("contrastizeFactor", 1.f, 0.01, 100, 0.01f, ImGuiSliderFlags_Logarithmic);
		blendWeaken = cfg2::getFloat("blendWeaken", 0.01f, 0.1, .499, .499f);
		weightFactor = cfg2::getFloat("weightFactor", 0.1f, 0.1, 30, 30, ImGuiSliderFlags_Logarithmic);

		if (pause2) {
			return;
		}
		img = multiscaleApply(img, [this](auto arg) { return update_1_scale(arg); });

		/*forxy(img) {
			auto& c = img(p);
			c = ci::constrain(c, 0.0f, 1.0f);
			auto c2 = 3.0f * c * c - 2.0f * c * c * c;
			c = mix(c, c2, contrastizeFactor);
			c = ci::constrain(c, 0.0f, 1.0f);
		}*/

	}
	void stefanDraw()
	{
		gl::setMatricesWindow(vec2(wsx, wsy), false);
		gl::clear(ColorA::black(), true);
		gl::disableDepthRead();


		sw::timeit("draw", [&]() {
			auto tex = gtex(img);
			gl::draw(redToLuminance(tex), getWindowBounds());
		});
	}
};
CrossThreadCallQueue* gMainThreadCallQueue;

CINDER_APP(SApp, RendererGl(),
	[&](ci::app::App::Settings* settings)
	{
		//bool developer = (bool)ifstream(getAssetPath("developer"));
		settings->setConsoleWindowEnabled(true);
		settings->setTitle("Volcano stuffs");
	})