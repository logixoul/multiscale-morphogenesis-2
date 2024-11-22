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
#include "util.h"

struct GpuScope {
	GpuScope(string name);
	~GpuScope();
};

#define GPU_SCOPE(name) GpuScope gpuScope_ ## __LINE__  (name);

void beginRTT(gl::TextureRef fbotex);
void endRTT();

void drawRect();

struct Str {
	string s;
	Str& operator<<(string s2) {
		s += s2 + "\n";
		return *this;
	}
	Str& operator<<(Str s2) {
		s += s2.s + "\n";
		return *this;
	}
	operator std::string() {
		return s;
	}
};

struct Uniform {
	function<void(gl::GlslProgRef)> setter;
	string shortDecl;
};

template<class T>
struct optional {
	T val;
	bool exists;
	optional(T const& t) { val=t; exists=true; }
	optional() { exists=false; }
};

template<class T> string typeToString();
template<> inline string typeToString<float>() {
	return "float";
}
template<> inline string typeToString<int>() {
	return "int";
}
template<> inline string typeToString<ivec2>() {
	return "ivec2";
}
template<> inline string typeToString<vec2>() {
	return "vec2";
}
struct ShadeOpts
{
	ShadeOpts();
	ShadeOpts& ifmt(GLenum val) { _ifmt=val; return *this; }
	ShadeOpts& scale(float val) { _scaleX=val; _scaleY=val; return *this; }
	ShadeOpts& scale(float valX, float valY) { _scaleX=valX; _scaleY=valY; return *this; }
	ShadeOpts& scope(std::string name) { _scopeName = name; return *this; }
	ShadeOpts& targetTex(gl::TextureRef val) { _targetTexs = { val }; return *this; }
	ShadeOpts& targetTexs(vector<gl::TextureRef> val) { _targetTexs = val; return *this; }
	ShadeOpts& targetImg(gl::TextureRef val) { _targetImg = val; return *this; }
	ShadeOpts& dstPos(ivec2 val) { _dstPos = val; return *this; }
	ShadeOpts& dstRectSize(ivec2 val) { _dstRectSize = val; return *this; }
	ShadeOpts& srcArea(Area val) {
		_area = val; return *this;
	}
	ShadeOpts& enableResult(bool val) {
		_enableResult = val; return *this;
	}
	template<class T>
	ShadeOpts& uniform(string name, T val) {
		_uniforms.push_back(Uniform{
			[val, name](gl::GlslProgRef prog) { prog->uniform(name, val); },
			typeToString<T>() + " " + name
			});
		return *this;
	}
	ShadeOpts& vshaderExtra(string val) {
		_vshaderExtra = val;
		return *this;
	}

	optional<GLenum> _ifmt;
	float _scaleX, _scaleY;
	std::string _scopeName;
	vector<gl::TextureRef> _targetTexs;
	gl::TextureRef _targetImg = nullptr;
	Area _area = Area::zero();
	ivec2 _dstPos;
	ivec2 _dstRectSize;
	bool _enableResult = true;
	vector<Uniform> _uniforms;
	string _vshaderExtra;
};

gl::TextureRef shade(vector<gl::TextureRef> const& texv, std::string const& fshader, ShadeOpts const& opts=ShadeOpts());
inline gl::TextureRef shade(vector<gl::TextureRef> const& texv, std::string const& fshader, float resScale);
