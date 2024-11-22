#include "precompiled.h"
//#include "ciextra.h"
#include "util.h"
//#include "shade.h"
#include "stuff.h"
#include "Array2D_imageProc.h"
#include "gpgpu.h"
#include "cfg2.h"
#include "sw.h"
#include "gpuBlur2_5.h"

#include "stefanfw.h"

#include "CrossThreadCallQueue.h"

typedef WrapModes::GetWrapped WrapMode;

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

// baseline 7fps
// now 9fps

//int wsx=800, wsy=800.0*(800.0/1280.0);
int wsx = 700, wsy = 700;
//int scale=2;
int sx = 256;
int sy = 256;
Array2D<float> img(sx, sy);
bool pause2 = false;
std::map<int, gl::TextureRef> texs;

// I have a `restoring_functionality_after_merge` branch where I attempt to merge supportlib from Tonemaster

template<class T, class FetchFunc>
Array2D<T> gauss3_(Array2D<T> src) {
	T zero = ::zero<T>();
	Array2D<T> dst1(src.w, src.h);
	Array2D<T> dst2(src.w, src.h);
	forxy(dst1)
		dst1(p) = .25f * (2 * FetchFunc::fetch(src, p.x, p.y) + FetchFunc::fetch(src, p.x - 1, p.y) + FetchFunc::fetch(src, p.x + 1, p.y));
	forxy(dst2)
		dst2(p) = .25f * (2 * FetchFunc::fetch(dst1, p.x, p.y) + FetchFunc::fetch(dst1, p.x, p.y - 1) + FetchFunc::fetch(dst1, p.x, p.y + 1));
	return dst2;
}

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
			img(p) = std::rand()/float(RAND_MAX);
		}
	}

	typedef Array2D<float> Img;
	Img update_1_scale(Img aImg)
	{
		auto img = aImg.clone();
		
		auto tex = gtex(img);
		gl::TextureRef gradientsTex;
		gradientsTex = get_gradients_tex_v2(tex, GL_REPEAT, GL_CLAMP_TO_EDGE);
		tex = shade2(tex, gradientsTex,
			"vec2 grad = fetch2(tex2);"
			"vec2 dir = perpLeft(safeNormalized(grad));"
			""
			"float val = fetch1();"
			"float valLeft = fetch1(tex, tc + tsize * dir);"
			"float valRight = fetch1(tex, tc - tsize * dir);"
			"float add = (val - (valLeft + valRight) * .5f);"
			"if(add < 0.0) add = 0;"
			"_out.r = val + add * abc;"
			, ShadeOpts().uniform("abc", abc),
			"vec2 perpLeft(vec2 v) {"
			"	return vec2(-v.y, v.x);"
			"}"
		);
		
		/*auto imgb = gauss3_<float, WrapModes::GetWrapped>(img);//gaussianBlur(img, 3);
		//img=imgb;
		forxy(img) {
			img(p) = lerp(img(p), imgb(p), .8f);
		}*/
		auto texb = tex;
		for (int i = 0; i < 3; i++) {
			texb->setWrap(GL_REPEAT, GL_CLAMP_TO_EDGE);
			texb = gauss3tex(texb);
		}
		
		tex = shade2(tex, texb,
			"float f = fetch1();"
			"float fb = fetch1(tex2);"
			"_out.r = mix(f, fb, .8f);"
		);
		img = gettexdata<float>(tex, GL_RED, GL_FLOAT);
		img = ::to01_Cut(img);

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
		forxy(img) {
			auto& c = img(p);
			c = ci::constrain(c, 0.0f, 1.0f);
			auto c2 = 3.0f * c * c - 2.0f * c * c * c;
			c = mix(c, c2, contrastizeFactor);
			c = ci::constrain(c, 0.0f, 1.0f);
		}
		return img;
	}
	Img multiscaleApply(Img src, function<Img(Img)> func) {
		int size = std::min(src.w, src.h);
		auto state = src.clone();
		vector<Img> scales;
		auto filter = ci::FilterGaussian();
		while (size > 1)
		{
			scales.push_back(state);
			state = ::resize(state, state.Size() / 2, filter);
			size /= 2;
		}
		vector<Img> origScales = scales;
		for(auto & s : origScales) s = s.clone();
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
			float w = 1.0f - pow(i / float(scales.size() - 1), 10.0f);
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
	void stefanUpdate() {
		abc = cfg2::getFloat("morphogenesis", .02, 0.068, 20, 2.945, ImGuiSliderFlags_Logarithmic);
		contrastizeFactor = cfg2::getFloat("contrastizeFactor", 1.f, 0.01, 100, 0.912, ImGuiSliderFlags_Logarithmic);
		blendWeaken = cfg2::getFloat("blendWeaken", 0.01f, 0.1, .499, .389f);

		if (pause2) {
			return;
		}
		img = multiscaleApply(img, [this](auto arg) { return update_1_scale(arg); });
		//img = update_1_scale(img);
	}

	void stefanDraw()
	{
		gl::clear(Color(0, 0, 0));
		cout << "frame# " << getElapsedFrames() << endl;

		//gl::setMatricesWindowPersp(vec2(sx, sy), 90.0f, 1.0f, 1000.0f, false);
		gl::setMatricesWindow(vec2(wsx, wsy), false);
		gl::clearDepth(0.0f);
		gl::clear(ColorA::black(), true);
		gl::disableDepthRead();
		//gl::enableDepth();


		sw::timeit("draw", [&]() {
			if (1) {
				//renderer.render(img);
				auto tex = gtex(img);
				auto grads = ::get_gradients_tex(tex);
				tex = shade2(tex, grads,
					"float val = fetch1();"
					"float fw = fwidth(val);"
					//"val = smoothstep(0.5-fw/2, 0.5+fw/2, val);"
					// this is taken from https://www.shadertoy.com/view/Mld3Rn
					"vec3 fire = vec3(min(val * 1.5, 1.), pow(val, 2.5), pow(val, 12.)); "
					"float der = fetch2(tex2).y;"
					"if(der > 0.03 && der < 0.2 && val > 0.1) der = der * 15.0 + .1;"
					"der = max(0, der);"
					//"fire += der;"
					"_out.rgb = fire;",
					ShadeOpts().ifmt(GL_RGB16F)
					);
				//tex = redToLuminance(tex);
				//tex->setMagFilter(GL_NEAREST);
				gl::draw(tex, getWindowBounds());
			}
			else {
				vector<gl::TextureRef> ordered;
				do {
					for(auto & pair : texs) {
						ordered.push_back(pair.second);
					}
				} while (0);

				float my = std::max(0.0f, std::min(1.0f, mouseY));
				int i = (texs.size() - 1) * my;
				auto tex = ordered[i];
				tex->bind();
				//tex.setMagFilter(GL_NEAREST);
				gl::draw(redToLuminance(tex), getWindowBounds());
			}
			});
	}
};
CrossThreadCallQueue* gMainThreadCallQueue;

CINDER_APP(SApp, RendererGl(),
	[&](ci::app::App::Settings* settings)
	{
		//bool developer = (bool)ifstream(getAssetPath("developer"));
		settings->setConsoleWindowEnabled(true);
	})