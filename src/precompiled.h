#pragma once
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

//#define BOOST_RESULT_OF_USE_DECLTYPE 
#include <cmath>
#include <iostream>
#include <string>
#include <complex>
#undef min
#undef max
#include <cinder/ip/Resize.h>
#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/GlslProg.h>
#include <cinder/gl/Texture.h>
#include <cinder/gl/Fbo.h>
#include <cinder/gl/gl.h>
#include <cinder/ImageIo.h> // todo rm?

//#include <cinder/Vector.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/gtx/io.hpp>
#define IMGUI_USER_CONFIG "CinderImGuiConfig.h"
#define CINDER_IMGUI_EXTERNAL

using namespace glm;
using namespace std;
using namespace ci;
using namespace ci::app;
using namespace std::experimental;
#include <glm/gtx/matrix_transform_2d.hpp>
#include <fftw3.h>

//#include <tuple>
#include <queue>
#include <opencv2/imgproc.hpp>
//#include <thread>


typedef gl::TextureRef Tex;