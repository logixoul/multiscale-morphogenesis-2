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
#include "TextureCache.h"

namespace gpuBlur2_5 {
	gl::TextureRef run(gl::TextureRef src, int lvls);
	gl::TextureRef run_longtail(gl::TextureRef src, int lvls, float lvlmul, float hscale = .5f, float vscale = .5f);
	float getGaussW();
	float gauss(float f, float width);
	gl::TextureRef upscale(gl::TextureRef src, ci::ivec2 toSize);
	gl::TextureRef upscale(gl::TextureRef src, float hscale, float vscale);
	gl::TextureRef singleblur(gl::TextureRef src, float hscale, float vscale, GLenum wrap = GL_CLAMP_TO_BORDER);
}

namespace gpuBlur = gpuBlur2_5;