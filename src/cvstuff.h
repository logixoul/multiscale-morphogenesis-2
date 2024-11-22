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

void mm(string name, cv::Mat mat);

void to01(cv::Mat& mat);

gl::TextureRef gtex(cv::Mat a, GLenum ifmt = 0, GLenum fmt=0);

cv::Mat dlToMat(gl::TextureRef tex, int mipLevel = 0);

cv::Mat dlToMat3(gl::TextureRef tex, int mipLevel = 0);

gl::TextureRef t(gl::TextureRef in);

void publish(string desc, cv::Mat mat);

void publish(string desc, gl::TextureRef tex);

extern void my_imshow(string desc, cv::Mat mat);