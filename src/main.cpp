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

auto vertShader = CI_GLSL(150,
	uniform mat4 ciModelViewProjection;
	uniform mat4 ciModelView;
	uniform mat3 ciNormalMatrix;

	in vec4 ciPosition;
	in vec4 ciColor;
	out lowp vec4 Color;
	in vec3 ciNormal;
	out highp vec3 Normal;
	out highp vec3 ViewPos;
	out highp vec3 LightPos;
	void main(void)
	{
		gl_Position = ciModelViewProjection * ciPosition;
		Color = ciColor;
		Normal = ciNormalMatrix * ciNormal;

		// Compute view-space position of the vertex (ChatGPT)
		ViewPos = vec3(ciModelView * ciPosition);
		const vec3 L = normalize(vec3(0, -.3, -1));

		LightPos = normalize(ciNormalMatrix * L);
	}
);

auto fragShader = CI_GLSL(150,
	out vec4 oColor;
	in vec4 Color;
	in vec3 Normal;
	in highp vec3 ViewPos;
	in highp vec3 LightPos;
	uniform samplerCube uCubeMapTex;
	void main(void)
	{
		vec3 V = normalize(-ViewPos); // Camera is at (0,0,0) in view space
		vec3 N = normalize(Normal);
		float lambert = max(0.0, dot(N, LightPos));
		//float specular = pow(max(0.0, dot(reflect(-LightPos, N), V)), 16.0);
		vec3 specular = texture(uCubeMapTex, reflect(-LightPos, N)).rgb;

		float fresnelBase = 0.1; // reflectance at normal incidence (F₀)
		float fresnel = fresnelBase + (1.0 - fresnelBase) * pow(1.0 - max(dot(N, V), 0.0), 5.0);

		oColor.rgb = Color.rgb * lambert + specular *fresnel*.3;
		oColor.a = 1.0;
	}
);

/*struct GetWrappedX {
	template<class T>
	static T& fetch(Array2D<T>& src, int x, int y)
	{
		int clamped
		return src(
		return ::getWrapped(src, x, y);
	}
};*/

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
	CameraPersp		mCam;
	CameraUi		mCamUi;
	gl::BatchRef	mPointsBatch;
	gl::VboMeshRef	mVboMesh;
	gl::TextureCubeMapRef	mCubeMap;
	
	void setup()
	{
		auto format = gl::TextureCubeMap::Format().mipmap().internalFormat(GL_RGBA16F);
		format.setDataType(GL_FLOAT);
		mCubeMap = gl::TextureCubeMap::create(loadImage(loadAsset("blue_photo_studio_4k.hdr")), format);


		reset();
		enableDenormalFlushToZero();
		setWindowSize(wsx, wsy);


		disableGLReadClamp();
		stefanfw::eventHandler.subscribeToEvents(*this);

		cfg2::init();

		// code from cinder_0.9.2_vc2015\samples\ImageHeightField\src
		mCamUi = ci::CameraUi(&mCam, getWindow());
        mCam.setNearClip(10);
		mCam.setFarClip(2000);

		int mWidth = sx;
		int mHeight = sy;


		auto shaderProg = gl::GlslProg::create(vertShader, fragShader);
		shaderProg->uniform("uCubeMapTex", 0);
		// * 6 because each quad is made of 2 triangles, and each triangle has 3 vertices
		mVboMesh = gl::VboMesh::create(mWidth * mHeight * 6, GL_TRIANGLES, { gl::VboMesh::Layout().usage(GL_STATIC_DRAW).attrib(geom::POSITION, 3).attrib(geom::NORMAL, 3).attrib(geom::COLOR, 3) });
		mPointsBatch = gl::Batch::create(mVboMesh, shaderProg);

		updateData(gtex(img));
		mCam.lookAt(vec3(0, 0, -mWidth * .5), vec3(0, 0, 0));
		mCam.setFov(70.0f);
	}

	struct VertInfo {
		vec3 pos;
		vec3 normal;
		vec3 color;
	};

	// done by chatgpt
	Array2D<vec3> calcNormals(Array2D<float> heightField) {
		int w = heightField.w;
		int h = heightField.h;
		Array2D<vec3> normals(w, h);

		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				ivec2 p(x, y);

				// Use clamped neighbors for borders
				ivec2 px0(std::max(x - 1, 0), y);
				ivec2 px1(std::min(x + 1, w - 1), y);
				ivec2 py0(x, std::max(y - 1, 0));
				ivec2 py1(x, std::min(y + 1, h - 1));

				vec3 left = calcVertPos(px0, heightField);
				vec3 right = calcVertPos(px1, heightField);
				vec3 down = calcVertPos(py0, heightField);
				vec3 up = calcVertPos(py1, heightField);

				// Central difference vectors
				vec3 dx = right - left;
				vec3 dz = up - down;

				// Normal from cross product
				vec3 normal = normalize(cross(dz, dx));
				normals(x, y) = normal;
			}
		}

		return normals;
	}
	vec3 calcVertPos(ivec2 p, Array2D<float> const& redImg)
	{
		float height = redImg(p);
		float x = p.x - redImg.w / 2.0f;
		float y = p.y - redImg.h / 2.0f;
		return vec3(x, y, height * 30.0f);
	}

	VertInfo getVertInfo(ivec2 p, Array2D<vec3> const& rgbImg, Array2D<vec3> const& normalsImg, Array2D<float> const& redImg)
	{
		float height = redImg(p);
		VertInfo vert;
		vert.pos = calcVertPos(p, redImg);
		vert.color = rgbImg(p);
		vert.normal = normalsImg(p);
		return vert;
	}

	void updateData(Tex redTex)
	{
		auto rgbTex = redToRgb(redTex);
		auto rgbImg = dl<vec3>(rgbTex);
		auto redImg = dl<float>(redTex);

		auto normalsImg = calcNormals(redImg);

		auto vertPosIter = mVboMesh->mapAttrib3f(geom::POSITION);
		auto vertNormalIter = mVboMesh->mapAttrib3f(geom::NORMAL);
		auto vertColorIter = mVboMesh->mapAttrib3f(geom::COLOR);

		forxy(img) {
			if(p.x == img.w-1 || p.y == img.h-1)
				continue;

			VertInfo vert00 = getVertInfo(p, rgbImg, normalsImg, redImg);
			VertInfo vert01 = getVertInfo(p + ivec2(0, 1), rgbImg, normalsImg, redImg);
			VertInfo vert10 = getVertInfo(p + ivec2(1, 0), rgbImg, normalsImg, redImg);
			VertInfo vert11 = getVertInfo(p + ivec2(1, 1), rgbImg, normalsImg, redImg);

			auto colorForVert = vert00.color;
			*vertPosIter++ = vert00.pos;
			*vertPosIter++ = vert01.pos;
			*vertPosIter++ = vert11.pos;
			*vertPosIter++ = vert00.pos;
			*vertPosIter++ = vert11.pos;
			*vertPosIter++ = vert10.pos;

			*vertNormalIter++ = vert00.normal;
			*vertNormalIter++ = vert01.normal;
			*vertNormalIter++ = vert11.normal;
			*vertNormalIter++ = vert00.normal;
			*vertNormalIter++ = vert11.normal;
			*vertNormalIter++ = vert10.normal;


			*vertColorIter++ = vert00.color;
			*vertColorIter++ = vert01.color;
			*vertColorIter++ = vert11.color;
			*vertColorIter++ = vert00.color;
			*vertColorIter++ = vert11.color;
			*vertColorIter++ = vert10.color;
		}

		vertPosIter.unmap();
		vertNormalIter.unmap();
		vertColorIter.unmap();
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
		//gradientsTex = get_gradients_tex_v2(tex, GL_REPEAT, GL_CLAMP_TO_EDGE);
		gradientsTex = get_gradients_tex_v2(tex, GL_REPEAT, GL_CLAMP_TO_EDGE);

		/*auto gradients = dl<vec2>(gradientsTex);
		static const auto perpLeft = [&](vec2 v) { return vec2(-v.y, v.x); }; //correct
		auto guidance = img;
		auto img2 = img.clone();
		for (int x = 0; x < img.w; x++)
		{
			for (int y = 0; y < img.h; y++)
			{
				vec2 p = vec2(x, y);
				vec2 grad = safeNormalized(gradients(x, y));

				vec2 gradP = perpLeft(grad);

				float val = guidance(x, y);
				float valLeft = getBilinear<float, WrapModes::GetWrapped>(guidance, p + gradP);
				float valRight = getBilinear<float, WrapModes::GetWrapped>(guidance, p - gradP);
				float add = (val - (valLeft + valRight) * .5f);
				if (add < 0.0)
					add = 0;
				aaPoint<float, WrapModes::GetWrapped>(img2, p - grad, add * abc);
				//img2(p) += add * abc;
			}
		}*/
		tex = shade2(tex, gradientsTex,
			"vec2 grad = fetch2(tex2);"
			"vec2 dir = perpLeft(safeNormalized(grad));"
			""
			"float val = fetch1();"
			"float valLeft = fetch1(tex, tc + tsize * dir);"
			"float valRight = fetch1(tex, tc - tsize * dir);"
			"float add = (val - (valLeft + valRight) * .5f);"
			//"if(add < 0.0) add = 0;"
			"_out.r = val + add * abc;"
			, ShadeOpts().uniform("abc", abc),
			"vec2 perpLeft(vec2 v) {"
			"	return vec2(-v.y, v.x);"
			"}"
		);
		/*auto tex3 = shade2(tex, "float f = fetch1();"
			"_out.r = f;"
			"_out.a = 1.0;",
			ShadeOpts().ifmt(GL_RGBA8)
		);
		ci::writeImage("dbg.png", tex3->createSource());
		quit();*/
		//quit();
		//tex = gtex(img2);
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
		//for(int i = 0; i< 4; i++)
		while (size > 2)
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
		abc = cfg2::getFloat("morphogenesis", .02, 0.068, 20, 1.369, ImGuiSliderFlags_Logarithmic);
		contrastizeFactor = cfg2::getFloat("contrastizeFactor", 1.f, 0.01, 100, 1.198f, ImGuiSliderFlags_Logarithmic);
		blendWeaken = cfg2::getFloat("blendWeaken", 0.01f, 0.1, .499, .499f);
		weightFactor = cfg2::getFloat("weightFactor", 0.1f, 0.1, 30, 30, ImGuiSliderFlags_Logarithmic);

		if (pause2) {
			return;
		}
		img = multiscaleApply(img, [this](auto arg) { return update_1_scale(arg); });
		//img = update_1_scale(img);

		forxy(img) {
			auto& c = img(p);
			c = ci::constrain(c, 0.0f, 1.0f);
			auto c2 = 3.0f * c * c - 2.0f * c * c * c;
			c = mix(c, c2, contrastizeFactor);
			c = ci::constrain(c, 0.0f, 1.0f);
		}

	}
	Tex redToRgb(Tex red) {
		auto grads = ::get_gradients_tex(red);
		return shade2(red, grads,
			"float val = fetch1();"
			"float fw = fwidth(val);"
			// this is taken from https://www.shadertoy.com/view/Mld3Rn
			"vec3 fire = vec3(min(val * 1.5, 1.), pow(val, 2.5), pow(val, 12.)); "
			"float der = fetch2(tex2).x;"
			//"fire = createRotationMatrix(vec3(1,1,1), der * 100.0) * fire;"
			"_out.rgb = fire;",
			ShadeOpts().ifmt(GL_RGB16F),
			// from chatgpt
			MULTILINE(
				mat3 createRotationMatrix(vec3 axis, float radians) {
					axis = normalize(axis);
					float c = cos(radians);
					float s = sin(radians);
					float oneMinusC = 1.0 - c;

					float x = axis.x;
					float y = axis.y;
					float z = axis.z;

					return mat3(
						c + x * x * oneMinusC, x * y * oneMinusC - z * s, x * z * oneMinusC + y * s,
						y * x * oneMinusC + z * s, c + y * y * oneMinusC, y * z * oneMinusC - x * s,
						z * x * oneMinusC - y * s, z * y * oneMinusC + x * s, c + z * z * oneMinusC
					);
				}
			)
		);
	}
	void stefanDraw()
	{
		gl::setMatricesWindow(vec2(wsx, wsy), false);
		gl::clear(ColorA::black(), true);
		gl::disableDepthRead();
		

		sw::timeit("draw", [&]() {
			if (1) {
				auto tex = gtex(img);
				updateData(tex);

				//gl::draw(tex, getWindowBounds());
				
				gl::enableAlphaBlending();
				gl::enableDepthRead();
				gl::enableDepthWrite();

				gl::pushMatrices();
				gl::setMatrices(mCam);
				gl::ScopedTextureBind texBind(mCubeMap, 0);
				if (mPointsBatch)
					mPointsBatch->draw();
				gl::popMatrices();
				gl::disableDepthRead();
				gl::disableDepthWrite();
				gl::disableAlphaBlending();
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
		settings->setTitle("Volcano stuffs");
	})