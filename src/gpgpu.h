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

#pragma once
#include "precompiled.h"
#include "shade.h"

gl::TextureRef get_gradients_tex(gl::TextureRef src, GLuint wrap = GL_REPEAT);

//gl::TextureRef gradientForwardTex(gl::TextureRef src, GLuint wrap = GL_REPEAT);
//gl::TextureRef divBackwardTex(gl::TextureRef src, GLuint wrap = GL_REPEAT);
gl::TextureRef baseshade2(vector<gl::TextureRef> texv, string const& src, ShadeOpts const& opts = ShadeOpts(), string const& lib = "");
gl::TextureRef shade2(
	gl::TextureRef tex,
	string const& src, ShadeOpts const& opts = ShadeOpts(), string const& lib = "");
gl::TextureRef shade2(
	gl::TextureRef tex, gl::TextureRef tex2,
	string const& src, ShadeOpts const& opts = ShadeOpts(), string const& lib = "");
gl::TextureRef shade2(
	gl::TextureRef tex, gl::TextureRef tex2, gl::TextureRef tex3,
	string const& src, ShadeOpts const& opts = ShadeOpts(), string const& lib = "");
gl::TextureRef shade2(
	gl::TextureRef tex, gl::TextureRef tex2, gl::TextureRef tex3, gl::TextureRef tex4,
	string const& src, ShadeOpts const& opts = ShadeOpts(), string const& lib = "");
gl::TextureRef shade2(
	gl::TextureRef tex, gl::TextureRef tex2, gl::TextureRef tex3, gl::TextureRef tex4, gl::TextureRef tex5,
	string const& src, ShadeOpts const& opts = ShadeOpts(), string const& lib = "");
gl::TextureRef shade2(
	gl::TextureRef tex, gl::TextureRef tex2, gl::TextureRef tex3, gl::TextureRef tex4, gl::TextureRef tex5, gl::TextureRef tex6,
	string const& src, ShadeOpts const& opts = ShadeOpts(), string const& lib = "");
gl::TextureRef gauss3tex(gl::TextureRef src);

gl::TextureRef get_laplace_tex(gl::TextureRef src, GLuint wrap);

gl::TextureRef dotTex(gl::TextureRef lhs, gl::TextureRef rhs);

float dot(gl::TextureRef lhs, gl::TextureRef rhs);

struct Operable {
	explicit Operable(gl::TextureRef aTex);
	Operable operator+(gl::TextureRef other);
	Operable operator-(gl::TextureRef other);
	Operable operator*(gl::TextureRef other);
	Operable operator/(gl::TextureRef other);
	/*Operable operator+(float scalar) {
		globaldict["scalar"] = scalar;
		return Operable(shade2(tex, "_out.r = fetch1() + scalar;"));
	}
	Operable operator-(float scalar) {
		globaldict["scalar"] = scalar;
		return Operable(shade2(tex, "_out.r = fetch1() - scalar;"));
	}
	Operable operator*(float scalar) {
		globaldict["scalar"] = scalar;
		return Operable(shade2(tex, "_out.r = fetch1() * scalar;"));
	}
	Operable operator/(float scalar) {
		globaldict["scalar"] = scalar;
		return Operable(shade2(tex, "_out.r = fetch1() / scalar;"));
	}*/
	void operator+=(gl::TextureRef other);
	void operator-=(gl::TextureRef other);
	void operator*=(gl::TextureRef other);
	void operator/=(gl::TextureRef other);
	float dot(gl::TextureRef other);
	gl::TextureRef dotTex(gl::TextureRef other);
	operator gl::TextureRef();
private:
	gl::TextureRef tex;
};

Operable op(gl::TextureRef tex);

