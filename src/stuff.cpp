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
#include "stuff.h"
#include "gpgpu.h"
#include "TextureCache.h"

int denormal_check::num;

// tried to have this as a static member (with thread_local) but I got errors. todo.
/*thread_local*/ static std::map<string,string> FileCache_db;

string FileCache::get(string filename) {
	if(FileCache_db.find(filename)== FileCache_db.end()) {
		//std::vector<unsigned char> buffer;
		auto dataSource = loadAsset(filename);
		auto buffer = dataSource->getBuffer();
		auto data = (char*)buffer->getData();
		string bufferStr(data, data + buffer->getSize());
		FileCache_db[filename]=bufferStr;
	}
	return FileCache_db[filename];
}

static void APIENTRY messageCallback(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam)
{
	//	if (type != GL_DEBUG_TYPE_ERROR)
		//	return;
	if (type == GL_DEBUG_TYPE_PUSH_GROUP || type == GL_DEBUG_TYPE_POP_GROUP)
		return;

	cout << "GL CALLBACK. Msg: " << message << endl;

	if (type == GL_DEBUG_TYPE_ERROR) {
		cout << endl;
	}
}

void enableGlDebugOutput() {
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(messageCallback, 0);
}

void my_assert_func(bool isTrue, string desc) {
	if (!isTrue) {
		cout << "assert failure: " << desc << endl;
		system("pause");
		throw std::runtime_error(desc.c_str());
	}
}

void bind(gl::TextureRef & tex) {
	glBindTexture(tex->getTarget(), tex->getId());
}
void bindTexture(gl::TextureRef & tex) {
	glBindTexture(tex->getTarget(), tex->getId());
}

void bindTexture(gl::TextureRef tex, GLenum textureUnit)
{
	glActiveTexture(textureUnit);
	bindTexture(tex);
	glActiveTexture(GL_TEXTURE0); // todo: is this necessary?
}

gl::TextureRef gtex(Array2D<float> a)
{
	gl::TextureRef tex = maketex(a.w, a.h, GL_R16F);
	bind(tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, a.w, a.h, GL_RED, GL_FLOAT, a.data);
	return tex;
}

gl::TextureRef gtex(Array2D<vec2> a)
{
	gl::TextureRef tex = maketex(a.w, a.h, GL_RG16F);
	bind(tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, a.w, a.h, GL_RG, GL_FLOAT, a.data);
	return tex;
}

gl::TextureRef gtex(Array2D<vec3> a)
{
	gl::TextureRef tex = maketex(a.w, a.h, GL_RGB16F);
	bind(tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, a.w, a.h, GL_RGB, GL_FLOAT, a.data);
	return tex;
}
gl::TextureRef gtex(Array2D<bytevec3> a)
{
	gl::TextureRef tex = maketex(a.w, a.h, GL_RGB8);
	bind(tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, a.w, a.h, GL_RGB, GL_UNSIGNED_BYTE, a.data);
	return tex;
}

gl::TextureRef gtex(Array2D<vec4> a)
{
	gl::TextureRef tex = maketex(a.w, a.h, GL_RGBA16F);
	bind(tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, a.w, a.h, GL_RGBA, GL_FLOAT, a.data);
	return tex;
}

gl::TextureRef gtex(Array2D<uvec4> a)
{
	gl::Texture::Format fmt;
	fmt.setInternalFormat(GL_RGBA32UI);
	gl::TextureRef tex = gl::Texture2d::create(a.w, a.h, fmt);
	bind(tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, a.w, a.h, GL_RGBA_INTEGER, GL_UNSIGNED_INT, a.data);
	return tex;
}

ivec2 clampPoint(ivec2 p, int w, int h)
{
	ivec2 wp = p;
	if (wp.x < 0) wp.x = 0;
	if (wp.x > w - 1) wp.x = w - 1;
	if (wp.y < 0) wp.y = 0;
	if (wp.y > h - 1) wp.y = h - 1;
	return wp;
}

int sign(float f)
{
	if (f < 0)
		return -1;
	if (f > 0)
		return 1;
	return 0;
}

float expRange(float x, float min, float max) {
	return exp(lerp(log(min), log(max), x));
}

float niceExpRangeX(float mouseX, float min, float max) {
	float x2 = sign(mouseX)*std::max(0.0f, abs(mouseX) - 40.0f / (float)App::get()->getWindowWidth());
	return sign(x2)*expRange(abs(x2), min, max);
}

float niceExpRangeY(float mouseY, float min, float max) {
	float y2 = sign(mouseY)*std::max(0.0f, abs(mouseY) - 40.0f / (float)App::get()->getWindowHeight());
	return sign(y2)*expRange(abs(y2), min, max);
}

void setWrapBlack(gl::TextureRef tex) {
	// I think the border color is transparent black by default. It doesn't hurt that it is transparent.
	bind(tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	float black[] = { 0.0f, 0.0f, 0.0f, 0.0f };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, black);
	//tex->setWrap(GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER);
}

void setWrap(gl::TextureRef tex, GLenum wrap) {
	// I think the border color is transparent black by default. It doesn't hurt that it is transparent.
	bind(tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
}

gl::TextureRef maketex(int w, int h, GLint ifmt, bool allocateMipmaps, bool clear) {
	TextureCacheKey key;
	key.ifmt = ifmt;
	key.size = ivec2(w, h);
	key.allocateMipmaps = allocateMipmaps;
	auto tex = TextureCache::instance()->get(key);
	if(clear) {
		beginRTT(tex);
		gl::clear(ColorA::black());
		endRTT();
	}
	return tex;
}

void checkGLError(string place)
{
	GLenum errCode;
	if ((errCode = glGetError()) != GL_NO_ERROR)
	{
		cout << "GL error 0x" << hex << errCode << dec << " at " << place << endl;
	}
	else {
		cout << "NO error at " << place << endl;
	}
}


void disableGLReadClamp() {
	glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
}

void enableDenormalFlushToZero() {
	_controlfp(_DN_FLUSH, _MCW_DN);
}

/*void drawAsLuminance(gl::TextureRef const& in, const Rectf &dstRect) {
	shade2(in,
		"_out.rgb = vec3(fetch1());"
		, ShadeOpts()
		.enableResult(false)
		.dstPos(dstRect.getUpperLeft())
		.dstRectSize(dstRect.getSize())
		.fetchUpsideDown(true)
	);
}*/

unsigned int ilog2(unsigned int val) {
	unsigned int ret = -1;
	while (val != 0) {
		val >>= 1;
		ret++;
	}
	return ret;
}

vec2 compdiv(vec2 const & v1, vec2 const & v2) {
	float a = v1.x, b = v1.y;
	float c = v2.x, d = v2.y;
	float cd = sq(c) + sq(d);
	return vec2(
		(a*c + b * d) / cd,
		(b*c - a * d) / cd);
}

void draw(gl::TextureRef const& tex, ci::Rectf const& bounds) {
	throw "this code hasn't been updated";
	/*gl::TextureRef tex2 = tex;
	if (tex->getInternalFormat() == GL_R16F) {
		tex2 = redToLuminance(tex);
	}
	tex2->setTopDown(true);
	gl::draw(tex2, bounds);*/
}

void drawBetter(gl::TextureRef &texture, const Area &srcArea, const Rectf &dstRect, gl::GlslProgRef glslArg)
{
	texture->setTopDown(true);
	auto ctx = gl::context();

	Rectf texRect = texture->getAreaTexCoords(srcArea);

	gl::ScopedVao vaoScp(ctx->getDrawTextureVao());
	//ScopedBuffer vboScp(ctx->getDrawTextureVbo());
	glBindTexture(texture->getTarget(), texture->getId());

	gl::GlslProgRef glsl;
	if (glslArg != nullptr) {
		glsl = glslArg;
	}
	else {
		glsl = gl::getStockShader(gl::ShaderDef().color().texture(texture));
	}
	gl::ScopedGlslProg glslScp(glsl);
	glsl->uniform("uTex0", 0);

	ctx->setDefaultShaderVars();
	ctx->drawArrays(GL_TRIANGLE_STRIP, 0, 4);

	texture->setTopDown(false);
}

void drawBetter(gl::TextureRef &texture, const Rectf &dstRect, gl::GlslProgRef glslArg)
{
	drawBetter(texture, texture->getBounds(), dstRect, glslArg);
}

void myGLFence()
{
	auto fence = gl::Sync::create();
	fence->clientWaitSync(GL_SYNC_FLUSH_COMMANDS_BIT, (GLuint64)(-1)); // timeout: about 200 years
}

vector<string> toStrings(vector<filesystem::path> paths) {
	vector<string> res;
	for (auto& path : paths) {
		res.push_back(path.string());
	}
	return res;
}

inline void denormal_check::begin_frame() {
	num = 0;
}

inline void denormal_check::check(float f) {
	if (f != 0 && fabsf(f) < numeric_limits<float>::min()) {
		// it's denormalized
		num++;
	}
}

inline void denormal_check::end_frame() {
	cout << "denormals detected: " << num << endl;
}

template<> Array2D<bytevec3> dl<bytevec3>(gl::TextureRef tex) {
	return gettexdata<bytevec3>(tex, GL_RGB, GL_UNSIGNED_BYTE);
}

template<> Array2D<float> dl<float>(gl::TextureRef tex) {
	return gettexdata<float>(tex, GL_RED, GL_FLOAT);
}

template<> Array2D<vec2> dl<vec2>(gl::TextureRef tex) {
	return gettexdata<vec2>(tex, GL_RG, GL_FLOAT);
}

template<> Array2D<vec3> dl<vec3>(gl::TextureRef tex) {
	return gettexdata<vec3>(tex, GL_RGB, GL_FLOAT);
}

template<> Array2D<vec4> dl<vec4>(gl::TextureRef tex) {
	return gettexdata<vec4>(tex, GL_RGBA, GL_FLOAT);
}
