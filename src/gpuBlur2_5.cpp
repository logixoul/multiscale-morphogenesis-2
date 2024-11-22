/*
Tonemaster - HDR software
Copyright (C) 2018, 2019, 2020 Stefan Monov <logixoul@gmail.com>

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/



#include "precompiled.h"
#include "shade.h"
#include "stuff.h"
#include "shade.h"
#include "gpgpu.h" // for shade2
#include "gpuBlur2_5.h"
#include "stefanfw.h"

namespace gpuBlur2_5 {

	gl::TextureRef run(gl::TextureRef src, int lvls) {
		auto state = shade2(src, "_out.rgb = fetch3();");

		for (int i = 0; i < lvls; i++) {
			state = singleblur(state, .5, .5);
		}
		state = upscale(state, src->getSize());
		return state;
	}

	float logAB(float a, float b) {
		return log(b) / log(a);
	}

	gl::TextureRef run_longtail(gl::TextureRef src, int lvls, float lvlmul, float hscale, float vscale) {
		vector<float> weights;
		float sumw = 0.0f;
		int maxHLvls = floor(logAB(1.0f / hscale, (float)src->getWidth()));
		int maxVLvls = floor(logAB(1.0f / vscale, (float)src->getHeight()));
		int maxLvls = std::min(maxHLvls, maxVLvls);
		lvls = std::min(lvls, maxLvls);
		for (int i = 0; i < lvls; i++) {
			float w = pow(lvlmul, float(i));
			w += 1.0f;
			weights.push_back(w);
			sumw += w;
		}
		for (float& w : weights) {
			w /= sumw;
		}
		/*cout << "Weights: ";
		for(float w: weights) {
			cout << w << ", ";
		}
		cout << "\n";*/
		vector<gl::TextureRef> zoomstates;
		zoomstates.push_back(src);
		zoomstates[0] = shade2(zoomstates[0],
			"_out.rgb = fetch3() * _mul;",
			ShadeOpts().uniform("_mul", 1.0f / sumw));
		for (int i = 0; i < lvls; i++) {
			auto newZoomstate = singleblur(zoomstates[i], hscale, vscale);
			zoomstates.push_back(newZoomstate);
			if (newZoomstate->getWidth() < 1 || newZoomstate->getHeight() < 1) throw runtime_error("too many blur levels");
		}
		for (int i = lvls - 1; i > 0; i--) {
			auto upscaled = upscale(zoomstates[i], zoomstates[i - 1]->getSize());
			float w = pow(lvlmul, float(i)); // tmp copypaste
			zoomstates[i - 1] = shade2(zoomstates[i - 1], upscaled,
				"vec3 acc = fetch3(tex);"
				"vec3 nextzoom = fetch3(tex2);"
				"vec3 c = acc + nextzoom * _mul;"
				"_out.rgb = c;"
				, ShadeOpts().uniform("_mul", w)
			);
		}
		return zoomstates[0];
	}
	float getGaussW() {
		// default value determined by trial and error
		return 0.75f;
	}
	float gauss(float f, float width) {
		return exp(-f * f / (width*width));
	}
	gl::TextureRef upscale(gl::TextureRef src, ci::ivec2 toSize) {
		return upscale(src, float(toSize.x) / src->getWidth(), float(toSize.y) / src->getHeight());
	}
	gl::TextureRef upscale(gl::TextureRef src, float hscale, float vscale) {
		//globaldict["gaussW"] = getGaussW();
		string lib =
			"float gauss(float f, float width) {"
			"	return exp(-f*f/(width*width));"
			"}"
			;
		string shader =
			"	float gaussW = 0.75f;"
			"	vec2 offset = vec2(GB2_offsetX, GB2_offsetY);"
			// here tc2 is half a texel TO THE TOP LEFT of the texel center. IT IS IN UV SPACE.
			"	vec2 tc2 = floor(tc * texSize) / texSize;"
			// here I make tc2 be the texel center
			"	tc2 += tsize / 2.0;"
			// frXY is in PIXEL SPACE. its x and y go from -.5 to .5
			"	vec2 frXY = (tc - tc2) * texSize;"
			"	float fr = (GB2_offsetX == 1.0) ? frXY.x : frXY.y;"
			"	vec3 aM1 = fetch3(tex, tc2 + (-1.0) * offset * tsize);"
			"	vec3 a0 = fetch3(tex, tc2 + (0.0) * offset * tsize);"
			"	vec3 aP1 = fetch3(tex, tc2 + (+1.0) * offset * tsize);"
			"	"
			"	float wM1=gauss(-1.0-fr, gaussW);"
			"	float w0=gauss(-fr, gaussW);"
			"	float wP1=gauss(1.0-fr, gaussW);"
			"	float sum=wM1+w0+wP1;"
			"	wM1/=sum;"
			"	w0/=sum;"
			"	wP1/=sum;"
			"	_out.rgb = wM1*aM1 + w0*a0 + wP1*aP1;";
		setWrapBlack(src);
		auto hscaled = shade2(src, shader,
			ShadeOpts()
				.scale(hscale, 1.0f)
				.uniform("GB2_offsetX", 1.0f)
				.uniform("GB2_offsetY", 0.0f)
			, lib);
		setWrapBlack(hscaled);
		auto vscaled = shade2(hscaled, shader,
			ShadeOpts()
				.scale(1.0f, vscale)
				.uniform("GB2_offsetX", 0.0f)
				.uniform("GB2_offsetY", 1.0f)
			, lib);
		return vscaled;
	}
	gl::TextureRef singleblur(gl::TextureRef src, float hscale, float vscale, GLenum wrap) {
		GPU_SCOPE("singleblur");
		//float gaussW = mouseY * 4 + .1;
		float gaussW = 4;
		//cout << "2020gauss=" << gaussW<<endl;
		
		/*float w0 = (mouseY - .5) * .01 + .9958f;
		float w1 = (1 - w0) /2;*/
		float w0 = gauss(0.0, gaussW);
		float w1 = gauss(1.0, gaussW);
		float w2 = gauss(2.0, gaussW);
		/*float w0 = 2;
		float w1 = 1;*/
		float sum = 2.0f*(w1+w2) + w0;
		w2 /= sum;
		w1 /= sum;
		w0 /= sum;
		stringstream weights;
		weights << fixed << "float w0=" << w0 << ", w1=" << w1 << ", w2=" << w2 << ";" << endl;
		//cout << "weights=" << weights.str() << endl;
		string shader =
			"vec2 offset = vec2(GB2_offsetX, GB2_offsetY);"
			"vec3 aM2 = fetch3(tex, tc + (-2.0) * offset * tsize);"
			"vec3 aM1 = fetch3(tex, tc + (-1.0) * offset * tsize);"
			"vec3 a0 = fetch3(tex, tc + (0.0) * offset * tsize);"
			"vec3 aP1 = fetch3(tex, tc + (+1.0) * offset * tsize);"
			"vec3 aP2 = fetch3(tex, tc + (+2.0) * offset * tsize);"
			""
			+ weights.str() +
			"_out.rgb = w2 * (aM2 + aP2) + w1 * (aM1 + aP1) + w0 * a0;";

		//setWrapBlack(src);
		//setWrap(src, wrap);
		setWrapBlack(src);
		auto hscaled = shade2(src, shader,
			ShadeOpts()
				.scale(hscale, 1.0f)
				.uniform("GB2_offsetX", 1.0f)
				.uniform("GB2_offsetY", 0.0f)
			);
		//setWrapBlack(hscaled);
		//setWrap(hscaled, wrap);
		setWrapBlack(hscaled);
		auto vscaled = shade2(hscaled, shader,
			ShadeOpts()
			.uniform("GB2_offsetX", 0.0f)
			.uniform("GB2_offsetY", 1.0f)
			.scale(1.0f, vscale));
		return vscaled;
	}

}
