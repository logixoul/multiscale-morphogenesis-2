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
#include "cvstuff.h"
#include "stuff.h"
#include "gpgpu.h" // for shade2
#include "CrossThreadCallQueue.h"

void mm(string name, cv::Mat mat) {
	double minn, maxx;
	cv::minMaxLoc(mat, &minn, &maxx);
	//if(minn != maxx)
	cout << "[" << name << "] min=" << minn << ", max=" << maxx << ", range = " << maxx - minn << endl;
}

void to01(cv::Mat & mat) {
	double minn, maxx;
	cv::minMaxLoc(mat, &minn, &maxx);
	double range = maxx - minn;
	mat -= cv::Scalar(minn, minn, minn);
	mat /= range;
}

gl::TextureRef gtex(cv::Mat a, GLenum ifmt, GLenum fmt)
{
	cv::Mat aT = a; // todo
	gl::Texture::Format fmtObj;
	if (ifmt == 0) ifmt = GL_R32F;
	if (fmt == 0) fmt = GL_FLOAT;
	fmtObj.setInternalFormat(ifmt); // TODO: R16F
	gl::TextureRef tex = maketex(aT.cols, aT.rows, fmtObj.getInternalFormat());
	bind(tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, aT.cols, aT.rows, GL_RED, fmt, aT.data);
	return tex;
}

static int matTypeFromTex(gl::TextureRef tex) {
	switch (tex->getInternalFormat()) {
	case GL_R32F: return CV_32F; break;
	case GL_RGB32F: return CV_32FC3; break;
	case GL_RGB8: return CV_8UC3; break;
	default: throw 0;
	}
}

cv::Mat dlToMat(gl::TextureRef tex, int mipLevel) {
	ivec2 sz = gl::Texture2d::calcMipLevelSize(mipLevel, tex->getWidth(), tex->getHeight());
	cv::Mat data = cv::Mat(sz.y, sz.x, ::matTypeFromTex(tex));

	bind(tex);
	glGetTexImage(GL_TEXTURE_2D, mipLevel, GL_RED, GL_FLOAT, data.data);

	return data;
}

cv::Mat dlToMat3(gl::TextureRef tex, int mipLevel) { // todo
	ivec2 sz = gl::Texture2d::calcMipLevelSize(mipLevel, tex->getWidth(), tex->getHeight());
	cv::Mat data = cv::Mat(sz.y, sz.x, CV_32FC3);

	bind(tex);
	glGetTexImage(GL_TEXTURE_2D, mipLevel, GL_BGR, GL_FLOAT, data.data);

	return data;
}

gl::TextureRef t(gl::TextureRef in) {
	auto res = maketex(in->getHeight(), in->getWidth(), in->getInternalFormat());
	shade2(in,
		"ivec2 fc = ivec2(gl_FragCoord.xy);"
		"_out.r = texelFetch(tex, fc.yx, 0).x;"
		, ShadeOpts().targetTex(res).dstRectSize(res->getSize())
	);
	return res;
}

void publish(string desc, cv::Mat mat) {
	auto tex = gtex(mat);
	shade2(tex, "_out.r = fetch1();", ShadeOpts().scope("[publish] " + desc));
}

void publish(string desc, gl::TextureRef tex) {
	shade2(tex, "_out.r = fetch1();", ShadeOpts().scope("[publish] " + desc));
}

extern CrossThreadCallQueue* gMainThreadCallQueue;

void my_imshow(string desc, cv::Mat mat)
{
	cv::Mat toShow;

	if (mat.channels() == 1) {
		cv::merge(vector<cv::Mat>{mat, -mat, mat*0}, toShow);
	}
	else {
		my_assert(mat.channels() == 3);
		toShow = mat.clone();
	}
	if (toShow.type() == CV_32FC3) {
		toShow.convertTo(toShow, CV_8UC3, 255.0f);
	}
	else {
		my_assert(toShow.type() == CV_8UC3);
	}
	auto toShow_copy = toShow.clone();
	gMainThreadCallQueue->pushCall([desc, toShow_copy]() {
		//cv::imshow(desc, toShow_copy);
		throw "not implemented";
	});
}
