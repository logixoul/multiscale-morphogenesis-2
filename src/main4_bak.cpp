#include "precompiled.h"
#if 0
#include "ciextra.h"
#include "util.h"
//#include <boost/typeof/typeof.hpp>
//#include <boost/assign.hpp>
#include "stuff.h"
#include "shade.h"
#include "gpgpu.h"
#include "gpuBlur2_4.h"
#include "cfg1.h"
#include "sw.h"

#include <float.h>

typedef Array2D<float> Image;
int wsx=800, wsy = 800 * (800.0f / 1280.0f);
int scale = 4;
int sx = wsx / ::scale;
int sy = wsy / ::scale;
Image img(sx, sy);
Array2D<vec2> velocity(sx, sy, vec2());




bool pause;


void updateConfig() {
}

struct SApp : App {
	Rectf area;
		
	void setup()
	{
		enableDenormalFlushToZero();

		area = Rectf(0, 0, (float)sx-1, (float)sy-1).inflated(vec2());

		
		
		
		disableGLReadClamp();
		stefanfw::eventHandler.subscribeToEvents(*this);
		setWindowSize(wsx, wsy);

		forxy(img)
		{
			//img(p) = ci::randFloat(); // NEWCODE
		}

		tmpEnergy = Array2D<vec2>(sx, sy);
	}
	void keyDown(KeyEvent e)
	{
		if(keys['r'])
		{
			std::fill(img.begin(), img.end(), 0.0f);
			std::fill(velocity.begin(), velocity.end(), vec2());
		}
		if(keys['p'] || keys['2'])
		{
			pause = !pause;
		}
	}
	vec2 direction;
	vec2 lastm;
	void mouseDrag(MouseEvent e)
	{
		mm();
	}
	void mouseMove(MouseEvent e)
	{
		mm();
	}
	void mm()
	{
		direction = getMousePos() - lastm;
		lastm = getMousePos();
	}
	Array2D<float> img3;
	Array2D<vec2> tmpEnergy3;
	Array2D<vec2> tmpEnergy;

	void toTmpEnergy()
	{
		tmpEnergy = Array2D<vec2>(sx, sy);
		forxy(tmpEnergy)
		{
			tmpEnergy(p) = velocity(p) * img(p);
		}
	}

	void fromTmpEnergy()
	{
		forxy(tmpEnergy)
		{
			if(img(p) != 0.0f)
				velocity(p) = tmpEnergy(p) / img(p);
		}
	}

	void moveMatter(ivec2 p, vec2 p2, float movedMatter, vec2 transferredEnergy, vec2 vec2)
	{
		aaPoint_i2(img3, p, -movedMatter);
		aaPoint2(img3, p2, movedMatter);
		
		aaPoint_i2(tmpEnergy3, p, -transferredEnergy);
		if(area.contains(vec2(p) + vec2))
			aaPoint2(tmpEnergy3, p2, transferredEnergy);
	}

	/*string tmsg;
	void tnext(string msg)
	{
		tmsg = msg;
	}*/
	void draw()
	{
		my_console::beginFrame();
		sw::beginFrame();
		static bool first = true;
		if(first) { globaldict["ltm"] = 0.0f; globaldict["mul"] = 1.0f; mouseY = .4f; }
		globaldict_default("mul2", 1.0f);
		first = false;

		mouseX = getMousePos().x / (float)wsx;
		//if(keys['x']) {
		//	globaldict["ltm"] = sgn(mouseX) * exp(lmap(abs(mouseX), 0.0f, .3f, log(.001f), log(10.0f)));
		//}
		/*if(keys['y']) {
			globaldict["mul"] = mouseY;
		}
		if(keys['u']) {
			globaldict["mul2"] = mouseY;
		}*/
		auto surfTensionThres=cfg1::getOpt("surfTensionThres", .5f,
			[&]() { return keys['6']; },
			[&]() { return expRange(mouseY, 0.1f, 50000.0f); });
		auto surfTension=cfg1::getOpt("surfTension", 1.0f,
			[&]() { return keys['7']; },
			[&]() { return expRange(mouseY, .0001f, 40000.0f); });
		auto gravity=cfg1::getOpt("gravity", .1f,//0.0f,//.1f,
			[&]() { return keys['8']; },
			[&]() { return expRange(mouseY, .0001f, 40000.0f); });
		mouseY = getMousePos().y / (float)wsy;
		auto incompressibilityCoef=cfg1::getOpt("incompressibilityCoef", 1.0f,
			[&]() { return keys['\\']; },
			[&]() { return expRange(mouseY, .0001f, 40000.0f); });

		gl::clear(Color(0, 0, 0));

		if(!pause)
		{
			forxy(velocity)
			{
				//velocity(p) += vec2(0.0f, gravity);
				tmpEnergy(p) += vec2(0.0f, gravity) * img(p);
			}

			sw::timeit("img&energy gauss3", [&]() {
			img=gauss3(img);
			tmpEnergy=gauss3(tmpEnergy);
			});

			///////////////////// NEWCODE
			/*
			auto imgb = gauss3(img);
			auto imgbDiff = Array2D<float>(sx, sy);
			forxy(imgb)
			{
				imgbDiff(p) = imgb(p) - img(p);
			}
			//forxy(imgb)
			*/
			/////////////////////

			auto img_b = img.clone();
			sw::timeit("surftension&incompressibility [blurs]", [&]() {
				//for(int i = 0; i < 4; i++)
				//	img_b = gauss3(img_b);
				img_b = gaussianBlur(img_b, 6*2+1);
			});
			sw::timeit("surftension&incompressibility [the rest]", [&]() {
				forxy(velocity)
				{
					auto& guidance = img_b;
					auto g = gradient_i(guidance, p);
					//g *= surfTension;

					/*if(guidance(p) < surfTensionThres)
					{
						g *= surfTension * (guidance(p) - surfTensionThres);
					}
					else
					{
						g *= guidance(p);
					}*/
					float mul = guidance(p);
					if(guidance(p) < surfTensionThres)
					{
						mul += -surfTension;
					}
					else
					{
						mul *= incompressibilityCoef;
					}
					g *= mul;
					g *= -1.0f;
					//g += surfTension * -g * (1.0f / (guidance(p) + 1.0f));

					tmpEnergy(p) += g; // 
				}
			});
			
			sw::timeit("processFluid", [&]() {
			int times=1;
			for(int i = 0; i < times; i++) {
				img3=Array2D<float>(sx, sy);
				tmpEnergy3=Array2D<vec2>(sx, sy, vec2());
				auto area2=Rectf(area);
				area2.x2 -= .01f;
				area2.y2 -= .01f;
				forxy(img)
				{
					if(img(p) == 0.0f)
						continue;
					
					vec2 vec = (tmpEnergy(p) / img(p)) / float(times);
	
					vec2 dstOrig = vec2(p) + vec;
					vec2 dst = area2.closestPoint(dstOrig);

					aaPoint2_fast(img3, dst, img(p));
	
					auto transferredEnergy = tmpEnergy(p);
					if(area2.contains(dstOrig))
						aaPoint2_fast(tmpEnergy3, dst, transferredEnergy);
				}
				img = img3;
				tmpEnergy = tmpEnergy3;
			}
			});
			
			/*forxy(tmpEnergy)
			{
				tmpEnergy(p) *= 0.0f;
			}*/

			if(mouseDown_[0])
			{
				vec2 scaledm = vec2(mouseX * (float)sx, mouseY * (float)sy);
				Area a(scaledm, scaledm);
				int r = 5; //sqrt(800);
				a.expand(r, r);
				for(int x = a.x1; x <= a.x2; x++)
				{
					for(int y = a.y1; y <= a.y2; y++)
					{
						vec2 v = vec2(x, y) - scaledm;
						float w = max(0.0f, 1.0f - v.length() / r);
						w = 3 * w * w - 2 * w * w * w;
						w=max(0.0f,w);
						img.wr(x, y) += 1.f * w * exp(-5.0f + 10.0f * mouseX);
					}
				}
			} else if(mouseDown_[2]) {
				mm();
				vec2 scaledm = vec2(mouseX * (float)sx, mouseY * (float)sy);
				Area a(scaledm, scaledm);
				int r = 15; //sqrt(800);
				a.expand(r, r);
				for(int x = a.x1; x <= a.x2; x++)
				{
					for(int y = a.y1; y <= a.y2; y++)
					{
						vec2 v = vec2(x, y) - scaledm;
						float w = max(0.0f, 1.0f - v.length() / r);
						w = 3 * w * w - 2 * w * w * w;
						if(img.wr(x, y) != 0.0f)
							tmpEnergy.wr(x, y) += /*(1.0f / img.wr(x, y)) * */ w * img.wr(x, y) * /*1.2f*/4.0f * direction / (float)scale;
					}
				}
			}

			//fromTmpEnergy();
		} // if ! pause
		sw::timeit("draw", [&]() {
			float limit = 65504 - 1;
			auto img5=img.clone();
			if(keys['9']) {
				forxy(img5) {
					img5(p) = tmpEnergy(p).length();
				}
			}
			if(keys['0']) {
				forxy(img5) {
					img5(p) = velocity(p).length() * sqrt(img(p));
				}
			}

			float currentMax = *std::max_element(img5.begin(), img5.end());
			if(currentMax > limit)
				cout << "clamping " << currentMax << " to " << limit << " (ratio " << (currentMax/limit) << ")" << endl;
			forxy(img5)
			{
				img5(p) = min(img5(p), limit);
			}
		
			auto tex = gtex(img5);

			/*tex = shade2(tex, "float f=fetch1(); vec3 c = vec3(1.0);"
				"c*=30.0*exp(-f*5.0*vec3(.8,.6,.5));"
				//"c += vec3(.5, .7, .9) * f * f * .2;"
				"_out=c;", ShadeOpts().ifmt(GL_RGB16F));*/
			auto grads = get_gradients_tex(tex);
			/*string getN =
				Str()
				<< "vec3 getN(sampler2D tex_) {"
				<< "	float aL = texture2D(tex_, tc + tsize * vec2(-1.0, 0.0)).x;"
				<< "	float aR = texture2D(tex_, tc + tsize * vec2(+1.0, 0.0)).x;"
				<< "	float aU = texture2D(tex_, tc + tsize * vec2(0.0, -1.0)).x;"
				<< "	float aD = texture2D(tex_, tc + tsize * vec2(0.0, +1.0)).x;"
				<< "	float dx = (aR - aL)/2.0;"
				<< "	float dy = (aD - aU)/2.0;"
				<< "	return normalize(vec3(-dx, -dy, -1.0));"
				<< "}";*/
			/*
			"vec3 N=normalize(cross("
			"	vec3(1.0,0.0,g.x),"
			"	vec3(1.0,g.y,0.0)"
			"));"
			"vec3 I=-normalize(vec3(tc.x-.5, tc.y-.5, 1.0));"
			"float eta=.9;"
			"vec3 R=refract(I, N, eta);"
			*/
			static auto envMap = gl::Texture(ci::loadImage("envmap2.png"));
			static auto gradientMap = gl::Texture(ci::loadImage("gradientmap.png"));
			globaldict["surfTensionThres"] = surfTensionThres;

			// the following gives us a sharp refracting boundary and also lets us check for ==0.0 where we reflect.
			/*tex = shade2(tex,
				"float f=fetch1();"
				"f-=surfTensionThres;"
				"if(f<0.0) f=0.0;"
				"_out=vec3(f);");*/

			auto laplacetex = get_laplace_tex(tex);

			laplacetex  = shade2(laplacetex,
				"float laplace = max(fetch1(tex), 0.0);"
				"_out = vec3(laplace);"
				);
			//auto tex2b = gpuBlur2_4::run(tex2, 2, 2); // this is the slow part.
			auto laplacetexB = gpuBlur2_4::run_longtail(laplacetex, 5, 1.0f);
			//gl::draw(laplacetexB, getWindowBounds());
			auto laplacetexSum = shade2(laplacetex, laplacetexB,
				"_out = fetch3(tex) + fetch3(tex2);"
				);

			string sh_invLumReinhard =
				"vec3 invLumReinhard(vec3 c) {"
				"	vec3 w = vec3(.22, .71, .07);"
				"	float lum = dot(c, w);"
				"	c /= lum;"
				"	c *= lum / (1.0-.99*lum);"
				"	return c;"
				"}";

			// gradient-map the bloom tex
			auto laplaceBGradientmapped = shade2(laplacetexSum, gradientMap,
				"float laplace = fetch1(tex);"
				"float lum=laplace;" // backcompat
				//"lum *= 5.0;" // so the brightest laplace places have lum 1 (otherwise they have lum 10 cause we multiply after the gradientfetch);
				"lum /= lum+1.0;"
				"lum = pow(lum, 2.0);"
				"_out = fetch3(tex2, vec2(lum, 0.0));" // fetch end of gradient (yellow)
				""
				//"_out = invLumReinhard(_out);", ShadeOpts(), sh_invLumReinhard
				);
			//gl::draw(laplaceBGradientmapped, getWindowBounds());
			
			auto tex2 = shade2(tex, grads, envMap, laplaceBGradientmapped,
				"vec2 grad = fetch2(tex2);"
				"vec3 N = normalize(vec3(-grad.x, -grad.y, -1.0));"
				"vec3 I=-normalize(vec3(tc.x-.5, tc.y-.5, 1.0));"
				"float eta=1.0/1.3;"
				"vec3 R=refract(I, N, eta);"
				"vec3 c = getEnv(R);"
				"vec3 albedo = 0.0*vec3(0.005, 0.0, 0.0);"
				"c = mix(albedo, c, pow(.9, fetch1(tex) * 50.0));" // tmp
				/*"R = reflect(I, N);"
				"if(fetch1(tex) > surfTensionThres)"
				"	c += getEnv(R) * .01;"*/
				
				//"c += vec3(1.0, 0.0, 0.0) * max(fetch1(tex4), 0.0) * 10.0;"
				"vec3 laplaceShadedB = fetch3(tex4);"
				"c += laplaceShadedB;"
				//"c += fetch1(tex) * vec3(1.0, 0.0, 0.0);"
				
				//"if(fetch1(tex) <= surfTensionThres)"
				//"	c = vec3(0.0);"
				"_out = c;"
				, ShadeOpts().ifmt(GL_RGB16F).scale(4.0f),
				"float PI = 3.14159265358979323846264;\n"
				"vec2 latlong(vec3 v) {\n"
				"v = v.xzy;\n"
				"v = normalize(v);\n"
				"float theta = acos(-v.z);\n" // +z is up
				"\n"
				"v.y=-v.y;\n"
				"float phi = atan(v.y, v.x) + PI;\n"
				//"return vec2(phi / (2.0*PI), theta / (PI/2.0));\n"
				"return vec2(phi / (2.0*PI), theta / (PI));\n"
				"}\n"
				"vec3 getEnv(vec3 v) {\n"
				"	vec3 c = fetch3(tex3, latlong(v));\n"
				//"	c = 5.0*pow(c, vec3(2.0));"
				"	c = pow(c, vec3(2.2));"
				//"	c/=vec3(1.0)-c; c*=1.0;"
				"	return c;"
				"}\n"
				);
			/*tex2 = shade2(tex2, tex2b,
				"_out = fetch3(tex) + fetch3(tex2);"
				);*/

			/*tex2 = shade2(tex2,
				"vec3 c = fetch3(tex);"
				"vec3 hcl = RGB2HCL(c);"
				"c = HCL2RGB(hcl);"
				"_out = c;"
				, ShadeOpts(), ntLoadFile("hcl_lib.fs"));*/

			if(0) tex2 = shade2(tex2,
				"vec3 c = fetch3();"
				"vec3 w = vec3(.22, .71, .07);"
				"float lum = dot(c, w);"
				"c /= lum;"
				"c *= lum / (lum+1.0);"
				"_out = c;"
				);

			// vignetting
			if(0) tex2 = shade2(tex2,
				"vec3 c = fetch3();"
				"vec2 tc2 = tc-vec2(.5);"
				//"c /= 1.0 + pow(length(tc2), 2.0);"
				"float ndist = length(tc2)/sqrt(.5*.5+.5*.5);" //normalized dist in [0,1]
				//"c *= pow(1.0-ndist, 3.0);"
				//"c *= exp(-ndist*ndist*12);"
				"float att = exp(-ndist*ndist*6);"
				"c /= c + vec3(1.0);"
				"c = pow(c, vec3(1.0/att));"
				"c /= vec3(1.0) - c;"
				"_out=c;"
				,ShadeOpts().scale(1.0f/::scale));

			// chromaclip
			/*tex2 = shade2(tex2,
				"vec3 c = fetch3();"
				"vec3 w = vec3(.22, .71, .07);"
				"float lum = dot(c, w);"
				"vec3 achrom=vec3(lum);"
				"c = mix(c, achrom, pow(lum/(lum+10.0),1.0));"
				"_out = c;"
				);*/

			// postbloom
			/*auto tex2b = gpuBlur2_4::run(tex2, 2, 1);
			tex2 = shade2(tex2, tex2b,
				"_out = fetch3(tex) + fetch3(tex2);"
				);*/

			tex2 = shade2(tex2,
				"vec3 c = fetch3(tex);"
				//"c = c / (c + vec3(1.0));"
				"c = pow(c, vec3(1.0/2.2));"
				/*
				"c.g = smoothstep(0.0, 1.0, c.g);"
				"c.b = max(0.0, min(1.0, c.b));"
				"c.b =1.0-pow(1.0-c.b,2.0);"
				"c.r = mix(.2, 1.0, c.r);"
				*/
				"_out = c;"
				);
			gl::draw(tex2, getWindowBounds());
			//testGpuBlur();
		});

#if 0
		auto tex6 = shade(list_of(gtex(img))(gtex(colormap)), "void shade() { /*float y=fetch1();*/"
			//"_out=fetch3(tex2,vec2(0.0, y));"
			"vec3 c=vec3(0.0);"
			"c.g = mix(0.1, 0.9, tc.y);"
			"float f=fetch1();"
			"c.r += c.g * (f/(f+1.0));"
			"c = pow(c, vec3(1.0/2.2));"
			"_out=c;"
			"}");
#endif
		/*auto tex6 = shade(list_of(gtex(img))(gtex(colormap)), "void shade() { float y=fetch1();"
			//"y /= y + 1.0;"
			"_out= min(1.0, y * 30.0) * fetch3(tex2,vec2(0.0, y));"
			"}");*/
		//auto tex6 = gtex(img);
		

		cout<<"==============="<<endl;
		cout<<"min: " << *std::min_element(img.begin(),img.end()) << ", ";
		cout<<"max: " << *std::max_element(img.begin(),img.end()) << endl;
		auto velBegin=(float*)velocity.begin();
		auto velEnd=(float*)velocity.end();
		cout<<"vmin: " << *std::min_element(velBegin,velEnd) << ", ";
		cout<<"vmax: " << *std::max_element(velBegin,velEnd) << endl;

		ivec2 scaledm2 = ivec2(mouseX * (float)sx, mouseY * (float)sy);
		auto toPrint = keys['g'] ? gradient_f(img, scaledm2) : gradient_i2(img, scaledm2);
		cout << "scale is " << scale << endl;
		cout << "gradient at p = " << toPrint << endl;
		cout << "surftension thres: " << surfTensionThres << endl;
		cout << "surface tension: " << surfTension << endl;
		cout << "gravity: " << gravity << endl;
		cout << "fps: " << getFrameRate() << endl;
		sw::endFrame();
		cfg1::print();
		my_console::endFrame();

		if(pause)
			Sleep(50);
		//Sleep(5);
#if 0
		if(getElapsedFrames()==1){
			if(!SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS/*BELOW_NORMAL_PRIORITY_CLASS*/))
			{
				throw 0;
			}
		}
#endif
	}
	// nullTerminatedLoadFile
	string ntLoadFile(string filename)
	{
		vector<unsigned char> vec;
		loadFile(vec, filename);
		return string(vec.begin(), vec.end());
	}
	// http://www.asawicki.info/news_1301_reflect_and_refract_functions.html
	vec3 refract(const vec3 &I, const vec3 &N, float eta)
	{
		float N_dot_I = ci::dot(N, I);
		float k = 1.f - eta * eta * (1.f - N_dot_I * N_dot_I);
		if (k < 0.f)
			return vec3(0.f, 0.f, 0.f);
		else
			return eta * I - (eta * N_dot_I + sqrtf(k)) * N;
	}
	void draw(gl::Texture tex, Rectf srcRect, Rectf dstRect)
	{
		glBegin(GL_QUADS);
		Rectf tc_(srcRect.getUpperLeft() / tex.getSize(), srcRect.getLowerRight() / tex.getSize());
		glTexCoord2f(tc_.getX1(), tc_.getY1()); glVertex2f(dstRect.getX1(), dstRect.getY1());
		glTexCoord2f(tc_.getX2(), tc_.getY1()); glVertex2f(dstRect.getX2(), dstRect.getY1());
		glTexCoord2f(tc_.getX2(), tc_.getY2()); glVertex2f(dstRect.getX2(), dstRect.getY2());
		glTexCoord2f(tc_.getX1(), tc_.getY2()); glVertex2f(dstRect.getX1(), dstRect.getY2());
		glEnd();
	}
	template<class T>
	vec2 gradient_f(Array2D<T> src, vec2 p)
	{
		vec2 gradient;
		gradient.x = getBilinear(src, p.x + 1, p.y) - getBilinear(src, p.x - 1, p.y);
		gradient.y = getBilinear(src, p.x, p.y + 1) - getBilinear(src, p.x, p.y - 1);
		return gradient;
	}
};

//#include <lib/define_winmain.h>


//CINDER_APP_BASIC(SApp, RendererGl)
int WINAPI WinMain(HINSTANCE hInstance,HINSTANCE hPrevInstance,LPSTR lpCmdLine,int nCmdShow) {	
	try{
		
		cinder::app::AppBasic::prepareLaunch();														
		cinder::app::AppBasic *app = new SApp;														
		cinder::app::Renderer *ren = new RendererGl;													
		cinder::app::AppBasic::executeLaunch( app, ren, "SApp" );										
		cinder::app::AppBasic::cleanupLaunch();														
		return 0;																					
	}catch(std::exception const& e) {
		cout << "caught: " << endl << e.what() << endl;
		//cout << "(exception type: " << typeinfo
		my_console::endFrame();
		//int dummy;cin>>dummy;
		system("pause");



	}
}

#endif