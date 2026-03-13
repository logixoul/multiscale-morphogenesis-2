#include "precompiled.h"
#include "util.h"
#include "stuff.h"
#include "Array2D_imageProc.h"
#include "gpgpu.h"
#include "cfg2.h"
#include "ThisSketch_ImageProcessingHelpers.h"
#include "stefanfw.h"
#include "CrossThreadCallQueue.h"
#include "gpuBlurClaude.h"

int wsx = 700, wsy = 700;

namespace ThisSketch {

	Array2D<float> img(256, 256);

	struct SApp : App {
		struct Options {
			float morphogenesisStrength;
			const float contrastizeStrength = 1.0f;
			float blendWeaken;
			float weightFactor;
			bool multiscale;
			bool binarizePostprocessing;
			bool pyramidOld;
			float highPassStrength;
			cfg2 cfg;

			void update() {
				cfg.begin();

				morphogenesisStrength = cfg.getFloat("morphogenesisStrength");
				blendWeaken = cfg.getFloat("blendWeaken");
				weightFactor = cfg.getFloat("weightFactor");
				multiscale = cfg.getBool("multiscale");
				binarizePostprocessing = cfg.getBool("binarizePostprocessing");
				pyramidOld = cfg.getBool("pyramidOld");
				highPassStrength = cfg.getFloat("highPassStrength");

				cfg.end();
			}
		};
		Options options;

		void setup()
		{
			reset();
			enableDenormalFlushToZero();
			setWindowSize(wsx, wsy);

			disableGLReadClamp();
			stefanfw::eventHandler.subscribeToEvents(*this);
		}

		void update()
		{
			options.update();
			stefanfw::beginFrame();
			stefanUpdate();
			stefanDraw();
			stefanfw::endFrame();
		}
		void keyDown(KeyEvent e)
		{
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
			auto img2 = img.clone();
			forxy(img) {
				vec2 const& pf = vec2(p);
				vec2 const& grad = gradients(p);
				vec2 const& gradN = safeNormalized(grad);
				vec2 const& gradNPerp = perpLeft(gradN);
				float add = -hessianDirectionalSecondDeriv<float, WrapModes::GetClamped>(img, p, gradNPerp);
				aaPoint<float, WrapModes::GetClamped>(img2, pf - gradN * add, add * options.morphogenesisStrength);
			}
			auto kernel = getGaussianKernel(3, sigmaFromKsize(3));
			//auto blurredImg2 = ::separableConvolve<float, WrapModes::GetClamped>(img2, kernel);
			auto blurredImg2 = ThisSketch::gaussianBlur3x3<float, WrapModes::GetClamped>(img2);
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
			return exp(options.weightFactor * iNormalized);
		}
		std::vector<float> getLevelWeights(int numLevels) const {
			std::vector<float> result;
			for (int i = 0; i < numLevels; i++) {
				result.push_back(getLevelWeight(i, numLevels));
			}
			float sum = std::accumulate(result.begin(), result.end(), 0.0f);
			for (auto& weight : result) {
				//	weight /= sum;
			}
			return result;
		}
		Img multiscaleApply(Img src, function<Img(Img)> func) {
			std::vector<Img> origScales = options.pyramidOld ? ThisSketch::buildGaussianPyramid_old(src) : ThisSketch::buildGaussianPyramid(src);
			std::vector<Img> updatedScales(origScales.size());
			static const auto filter = ci::FilterGaussian();
			const int last = origScales.size() - 1;
			updatedScales[last] = func(origScales[last]);
			auto weights = getLevelWeights(origScales.size());
			for (int i = updatedScales.size() - 1; i >= 1; i--) {
				auto diff = subtract(updatedScales[i], origScales[i]);
				diff = multiply(diff, weights[i]);
				auto upscaledDiff = ThisSketch::resize(diff, origScales[i - 1].Size(), filter);
				auto& nextScale = updatedScales[i - 1];
				nextScale = add(origScales[i - 1], upscaledDiff);
				nextScale = func(nextScale);
			}
			return updatedScales[0];
		}
		void stefanUpdate() {
			if (options.multiscale)
				img = multiscaleApply(img, [this](auto arg) { return updateSingleScale(arg); });
			else
				img = updateSingleScale(img);
			img = to01(img);
		}

		static gl::TextureRef gpuHighpass(gl::TextureRef in, float strength) {
			auto blurred = gpuBlurClaude::blurWithInvKernel(in);
			auto highpassed = shade2(in, blurred, MULTILINE(
				float f = fetch1();
			float fBlurred = fetch1(tex2);
			float highPassed = f - fBlurred * highPassStrength;
			_out.r = highPassed;
				), ShadeOpts().uniform("highPassStrength", strength)
				);
			return highpassed;
		}
		gl::TextureRef postprocess() {
			auto imgClamped = img.clone();
			forxy(imgClamped) imgClamped(p) = ci::constrain(imgClamped(p), 0.0f, 1.0f);

			auto imgTex = gtex(imgClamped);
			auto imgTexCentered = shade2(imgTex,
				"float f = fetch1();"
				"_out.r = f - .5;"
			);

			auto imgTexHighpassed = gpuHighpass(imgTexCentered, options.highPassStrength);
			imgTexHighpassed = gpuHighpass(imgTexHighpassed, options.highPassStrength);
			auto imgHighpassed = dl<float>(imgTexHighpassed);

			auto pyramid = buildGaussianPyramid(imgHighpassed);
			auto stateTex = maketex(wsx, wsy, GL_R16F, false, true);
			for (int i = pyramid.size() - 1; i >= 0; i--) {
				auto& thisLevel = pyramid[i];
				auto thisLevelTex = gtex(thisLevel);
				auto thisLevelTexContrastized = shade2(thisLevelTex,
					"float f = fetch1();"
					"float fw = fwidth(f);"
					"f = smoothstep(-fw/2.0, fw/2.0, f);"
					"_out.r = f;", ShadeOpts().dstRectSize(ivec2(wsx, wsy)));
				stateTex = op(stateTex) + thisLevelTexContrastized;
			}
			stateTex = op(stateTex) / float(pyramid.size());
			//stateTex = (op(stateTex) + op(gpuBlur2_5::run(stateTex, 3))*2.0f) / 2;
			stateTex = shade2(stateTex, MULTILINE(
				float val = fetch1();
			vec3 fire = vec3(min(val * 1.5, 1.), pow(val, 2.5), pow(val, 12.));
			_out.rgb = fire;
				),
				ShadeOpts().ifmt(GL_RGBA16F));
			return stateTex;
		}
		void stefanDraw()
		{
			gl::setMatricesWindow(vec2(wsx, wsy), false);
			gl::clear(ColorA::black(), true);
			gl::disableDepthRead();

			gl::TextureRef tex = gtex(img);
			if (options.binarizePostprocessing) {
				tex = postprocess();
			}
			else {
				tex = redToLuminance(tex);
			}
			gl::draw(tex, getWindowBounds());
		}
	};

}

CrossThreadCallQueue* gMainThreadCallQueue;

CINDER_APP(ThisSketch::SApp, RendererGl(),
	[&](ci::app::App::Settings* settings)
	{
		settings->setConsoleWindowEnabled(true);
	})