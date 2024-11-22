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
#include "gpgpu.h"
#include "stuff.h"

gl::TextureRef get_gradients_tex(gl::TextureRef src, GLuint wrap) {
	GPU_SCOPE("get_gradients_tex");
	glActiveTexture(GL_TEXTURE0);
	::bindTexture(src);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
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

/*gl::TextureRef gradientForwardTex(gl::TextureRef src, GLuint wrap) {
	GPU_SCOPE("gradientForwardTex");
	//src->setWrap(wrap, wrap);
	glActiveTexture(GL_TEXTURE0);
	::bindTexture(src);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
	return shade(list_of(src),
		"void shade(){"
		"	float srcHere=fetch1(tex,tc);"
		"	float srcR=fetch1(tex,tc+tsize*vec2(1.0,0.0));"
		"	float srcB=fetch1(tex,tc+tsize*vec2(0.0,1.0));"
		"	float dx=(srcR-srcHere)/2.0;"
		"	float dy=(srcB-srcHere)/2.0;"
		"	_out.xy=vec2(dx,dy);"
		"}"
		, ShadeOpts().ifmt(GL_RG16F));
}

gl::TextureRef divBackwardTex(gl::TextureRef src, GLuint wrap) {
	GPU_SCOPE("divBackwardTex");
	//src->setWrap(wrap, wrap);
	glActiveTexture(GL_TEXTURE0);
	::bindTexture(src);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
	return shade(list_of(src),
		"void shade(){"
		"	vec2 srcHere=fetch2(tex,tc);"
		"	vec2 srcL=fetch2(tex,tc-tsize*vec2(1.0,0.0));"
		"	vec2 srcU=fetch2(tex,tc-tsize*vec2(0.0,1.0));"
		"	float dx=(srcHere.x-srcL.x)/2.0;"
		"	float dy=(srcHere.y-srcU.y)/2.0;"
		"	_out.x = dx + dy;"
		"}",
		ShadeOpts().ifmt(GL_R16F)
	);
}*/

inline gl::TextureRef baseshade2(vector<gl::TextureRef> texv, string const& src, ShadeOpts const & opts, string const& lib)
{
	return shade(texv, lib + "void shade() {" + src + "}", opts);
}

gl::TextureRef shade2(gl::TextureRef tex, string const& src, ShadeOpts const & opts, string const& lib)
{
	return baseshade2({ tex }, src, opts, lib);
}

gl::TextureRef shade2(gl::TextureRef tex, gl::TextureRef tex2, string const& src, ShadeOpts const & opts, string const& lib)
{
	return baseshade2({ tex,tex2 }, src, opts, lib);
}

gl::TextureRef shade2(gl::TextureRef tex, gl::TextureRef tex2, gl::TextureRef tex3, string const& src, ShadeOpts const & opts, string const& lib)
{
	return baseshade2({ tex, tex2, tex3 }, src, opts, lib);
}

gl::TextureRef shade2(gl::TextureRef tex, gl::TextureRef tex2, gl::TextureRef tex3, gl::TextureRef tex4, string const& src, ShadeOpts const & opts, string const& lib)
{
	return baseshade2({ tex, tex2, tex3, tex4 }, src, opts, lib);
}

gl::TextureRef shade2(gl::TextureRef tex, gl::TextureRef tex2, gl::TextureRef tex3, gl::TextureRef tex4, gl::TextureRef tex5, string const& src, ShadeOpts const & opts, string const& lib)
{
	return baseshade2({ tex, tex2, tex3, tex4, tex5 }, src, opts, lib);
}

gl::TextureRef shade2(gl::TextureRef tex, gl::TextureRef tex2, gl::TextureRef tex3, gl::TextureRef tex4, gl::TextureRef tex5, gl::TextureRef tex6, string const& src, ShadeOpts const & opts, string const& lib)
{
	return baseshade2({ tex, tex2, tex3, tex4, tex5, tex6 }, src, opts, lib);
}

gl::TextureRef gauss3tex(gl::TextureRef src) {
	auto state = shade2(src,
		"vec3 sum = vec3(0.0);"
		"sum += fetch3(tex, tc + tsize * vec2(-1.0, -1.0)) / 16.0;"
		"sum += fetch3(tex, tc + tsize * vec2(-1.0, 0.0)) / 8.0;"
		"sum += fetch3(tex, tc + tsize * vec2(-1.0, +1.0)) / 16.0;"

		"sum += fetch3(tex, tc + tsize * vec2(0.0, -1.0)) / 8.0;"
		"sum += fetch3(tex, tc + tsize * vec2(0.0, 0.0)) / 4.0;"
		"sum += fetch3(tex, tc + tsize * vec2(0.0, +1.0)) / 8.0;"

		"sum += fetch3(tex, tc + tsize * vec2(+1.0, -1.0)) / 16.0;"
		"sum += fetch3(tex, tc + tsize * vec2(+1.0, 0.0)) / 8.0;"
		"sum += fetch3(tex, tc + tsize * vec2(+1.0, +1.0)) / 16.0;"
		"_out.rgb = sum;"
		);
	return state;
}

gl::TextureRef get_laplace_tex(gl::TextureRef src, GLuint wrap) {
	glActiveTexture(GL_TEXTURE0);
	::bindTexture(src);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
	auto state = shade2(src,
		"vec3 sum = vec3(0.0);"
		"sum += fetch3(tex, tc + tsize * vec2(-1.0, 0.0)) * -1.0;"
		"sum += fetch3(tex, tc + tsize * vec2(0.0, -1.0)) * -1.0;"
		"sum += fetch3(tex, tc + tsize * vec2(0.0, +1.0)) * -1.0;"
		"sum += fetch3(tex, tc + tsize * vec2(+1.0, 0.0)) * -1.0;"
		"sum += fetch3(tex, tc + tsize * vec2(0.0, 0.0)) * 4.0;"
		"_out.rgb = -sum / 4.0f;"
		);
	return state;
}
static int nextPowerOf2(int v) {
	float Lf = std::log2(v);
	if (Lf == floor(Lf)) {
		return v;
	}
	else {
		return 1 << int(std::ceil(Lf));
	}
}

gl::TextureRef pad(gl::TextureRef in, ivec2 newSize) {
	auto res = maketex(newSize.x, newSize.y, in->getInternalFormat(), true);
	//res = shade2(res, "_out = vec4(0);");
	beginRTT(res);
	gl::clear(ColorA::black(), false);
	{
		gl::ScopedViewport sv(ivec2(0, 0), in->getSize());
		gl::pushMatrices();
		{
			gl::setMatricesWindow(ivec2(1, 1), false);

			drawBetter(in, Rectf(0, 0, 1, 1));
			gl::popMatrices();
		}
	}
	endRTT();
	return res;
}
gl::TextureRef sumTex(gl::TextureRef in) {
	// todo: since the POT code here is commented out, things may be broken.
	// Without it, we have "breaking at iter 18" (0.33spf). With it, we have "breaking at iter 17" (~0.9spf)
	auto origSize = in->getSize();
	ivec2 sz = in->getSize(); sz.x = nextPowerOf2(sz.x); sz.y = nextPowerOf2(sz.y);
	in = pad(in, sz);
	bindTexture(in);
	glGenerateMipmap(in->getTarget());
	int mips = gl::Texture2d::requiredMipLevels(in->getWidth(), in->getHeight(), 1);
	auto res = maketex(1, 1, in->getInternalFormat());
	shade2(in, "_out.r = texelFetch(tex, ivec2(0,0), lastmip).x * area;"
		, ShadeOpts().targetTex(res).dstRectSize(res->getSize())
		.uniform("area", sz.x * sz.y)
		.uniform("lastmip", mips - 1)
	);
	return res;
}


gl::TextureRef dotTex(gl::TextureRef lhs, gl::TextureRef rhs) {
	auto muld = maketex(lhs->getWidth(), lhs->getHeight(), lhs->getInternalFormat(), false);
	shade2(lhs, rhs,
		"_out.r = fetch1() * fetch1(tex2);",
		ShadeOpts()
		.scope("mul")
		.targetTex(muld)
		.dstRectSize(muld->getSize())
	);
	return sumTex(muld);
}

float dot(gl::TextureRef lhs, gl::TextureRef rhs) {
	auto muld = dotTex(lhs, rhs);
	return dl<float>(muld)(0, 0);
}

Operable op(gl::TextureRef tex) {
	return Operable(tex);
}



inline Operable::Operable(gl::TextureRef aTex) {
	tex = aTex;
}

Operable Operable::operator+(gl::TextureRef other) {
	return Operable(shade2(tex, other, "_out.r = fetch1() + fetch1(tex2);"));
}
Operable Operable::operator-(gl::TextureRef other) {
	return Operable(shade2(tex, other, "_out.r = fetch1() - fetch1(tex2);"));
}
Operable Operable::operator*(gl::TextureRef other) {
	return Operable(shade2(tex, other, "_out.r = fetch1() * fetch1(tex2);"));
}
Operable Operable::operator/(gl::TextureRef other) {
	return Operable(shade2(tex, other, "_out.r = fetch1() / fetch1(tex2);"));
}

void Operable::operator+=(gl::TextureRef other) {
	tex = shade2(tex, other, "_out.r = fetch1() + fetch1(tex2);");
}
void Operable::operator-=(gl::TextureRef other) {
	tex = shade2(tex, other, "_out.r = fetch1() - fetch1(tex2);");
}
void Operable::operator*=(gl::TextureRef other) {
	tex = shade2(tex, other, "_out.r = fetch1() * fetch1(tex2);");
}
void Operable::operator/=(gl::TextureRef other) {
	tex = shade2(tex, other, "_out.r = fetch1() / fetch1(tex2);");
}
float Operable::dot(gl::TextureRef other) {
	return ::dot(tex, other);
}
gl::TextureRef Operable::dotTex(gl::TextureRef other) {
	return ::dotTex(tex, other);
}
Operable::operator gl::TextureRef() {
	return tex;
}