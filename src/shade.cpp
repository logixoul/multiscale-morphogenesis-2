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
#include "stefanfw.h"

/*thread_local*/ bool fboBound = false;

void beginRTT(gl::TextureRef fbotex)
{
	/*thread_local*/ static unsigned int fboid = 0;

	if(fboid == 0)
	{
		glGenFramebuffersEXT(1, &fboid);
	}
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboid);
	if (fbotex != nullptr)
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, fbotex->getId(), 0);
	fboBound = true;
}
void beginRTT(vector<gl::TextureRef> fbotexs)
{
	if (fbotexs.size() != 1)
		throw runtime_error("not implemented");

	/*thread_local*/ static unsigned int fboid = 0; // hack: a separate fbo for this case. this is brittle. todo.

	if (fboid == 0)
	{
		glGenFramebuffersEXT(1, &fboid);
	}
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboid);
	int i = 0;
	for (auto& fbotex : fbotexs) {
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT + i, GL_TEXTURE_2D, fbotex->getId(), 0);
		i++;
	}
	fboBound = true;
}
void endRTT()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	fboBound = false;
}

void drawRect() {
	auto ctx = gl::context();

	ctx->getDrawTextureVao()->bind();
	//ctx->getDrawTextureVbo()->bind(); // this seems to be unnecessary

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

auto samplerSuffix = [&](int i) -> string {
	return (i == 0) ? "" : std::to_string(1 + i);
};

auto samplerName = [&](int i) -> string {
	return "tex" + samplerSuffix(i);
};

std::string getCompleteFshader(vector<gl::TextureRef> const& texv, vector<Uniform> const& uniforms, std::string const& fshader, string* uniformDeclarationsRet) {
	auto texIndex = [&](gl::TextureRef t) {
		return std::to_string(
			1 + (std::find(texv.begin(), texv.end(), t) - texv.begin())
			);
	};
	stringstream uniformDeclarations;
	int location = 0;
	uniformDeclarations << "uniform ivec2 viewportSize;\n";
	uniformDeclarations << "uniform vec2 mouse;\n";
	//uniformDeclarations << "uniform vec2 resultSize;\n";
	//uniformDeclarations << "vec2 my_FragCoord;\n";
	for(int i = 0; i < texv.size(); i++)
	{
		string samplerType = "sampler2D";
		GLenum fmt, type;
		texv[i]->getInternalFormatInfo(texv[i]->getInternalFormat(), &fmt, &type);
		if (type == GL_UNSIGNED_INT) samplerType = "usampler2D";
		uniformDeclarations << "uniform " + samplerType + " " + samplerName(i) + ";\n";
		uniformDeclarations << "uniform vec2 " + samplerName(i) + "Size;\n";
		uniformDeclarations << "uniform vec2 tsize" + samplerSuffix(i) + ";\n";
	}
	for (auto& p : uniforms)
	{
		uniformDeclarations << "uniform " + p.shortDecl + ";\n";
	}
	//uniformDeclarations << "layout(binding=0, r32f) uniform coherent image2D image;";
	*uniformDeclarationsRet = uniformDeclarations.str();
	string intro =
		Str()
		<< "#version 150"
		<< "#extension GL_ARB_explicit_uniform_location : enable"
		<< "#extension GL_ARB_texture_gather : enable"
		//<< "#extension GL_ARB_shader_image_load_store : enable"
		<< uniformDeclarations.str()
		<< "in vec2 tc;"
		<< "in highp vec2 relOutTc;"
		<< "/*precise*/ out vec4 _out;"
		//<< "layout(origin_upper_left) in vec4 gl_FragCoord;"
		;
	intro += Str()
		<< "vec4 fetch4(sampler2D tex_, vec2 tc_) {"
		<< "	return texture2D(tex_, tc_).rgba;"
		<< "}"
		<< "vec3 fetch3(sampler2D tex_, vec2 tc_) {"
		<< "	return texture2D(tex_, tc_).rgb;"
		<< "}"
		<< "vec2 fetch2(sampler2D tex_, vec2 tc_) {"
		<< "	return texture2D(tex_, tc_).rg;"
		<< "}"
		<< "float fetch1(sampler2D tex_, vec2 tc_) {"
		<< "	return texture2D(tex_, tc_).r;"
		<< "}"
		<< "vec4 fetch4(sampler2D tex_) {"
		<< "	return texture2D(tex_, tc).rgba;"
		<< "}"
		<< "vec3 fetch3(sampler2D tex_) {"
		<< "	return texture2D(tex_, tc).rgb;"
		<< "}"
		<< "vec2 fetch2(sampler2D tex_) {"
		<< "	return texture2D(tex_, tc).rg;"
		<< "}"
		<< "float fetch1(sampler2D tex_) {"
		<< "	return texture2D(tex_, tc).r;"
		<< "}"
		<< "vec4 fetch4() {"
		<< "	return texture2D(tex, tc).rgba;"
		<< "}"
		<< "vec3 fetch3() {"
		<< "	return texture2D(tex, tc).rgb;"
		<< "}"
		<< "vec2 fetch2() {"
		<< "	return texture2D(tex, tc).rg;"
		<< "}"
		<< "float fetch1() {"
		<< "	return texture2D(tex, tc).r;"
		<< "}"
		<< "vec2 safeNormalized(vec2 v) { return length(v)==0.0 ? v : normalize(v); }"
		<< "vec3 safeNormalized(vec3 v) { return length(v)==0.0 ? v : normalize(v); }"
		<< "vec4 safeNormalized(vec4 v) { return length(v)==0.0 ? v : normalize(v); }"
		<< "#line 0\n\n" // the \n\n is needed only on Intel gpus. Probably a driver bug.
		;
	string outro =
		Str()
		<< "void main()"
		<< "{"
		<< "	_out = vec4(0.0f, 0.0f, 0.0f, 1.0f);"
		<< "	shade();"
		<< "}";
	return intro + fshader + outro;
}

// location++ breaks things on Intel
/*void setUniform(int location, float val) {
	glUniform1f(location, val);
}

void setUniform(int location, vec2 val) {
	glUniform2f(location, val.x, val.y);
}

void setUniform(int location, ivec2 val) {
	glUniform2i(location, val.x, val.y);
}

void setUniform(int location, int val) {
	glUniform1i(location, val);
}*/

gl::TextureRef shade(vector<gl::TextureRef> const& texv, std::string const& fshader, ShadeOpts const& opts)
{
	shared_ptr<GpuScope> gpuScope;
	if (opts._scopeName != "") {
		gpuScope = make_shared<GpuScope>(opts._scopeName);
	}
	//const string fshader = "void shade() { }";
	static std::mutex mapMutex;
	unique_lock<std::mutex> ul(mapMutex);
	static std::map<string, gl::GlslProgRef> shaders;
	gl::GlslProgRef shader;
	if(shaders.find(fshader) == shaders.end())
	{
		string uniformDeclarations;
		std::string completeFshader = getCompleteFshader(texv, opts._uniforms, fshader, &uniformDeclarations);
		string completeVshader = Str()
			<< "#version 150"
			<< "#extension GL_ARB_explicit_uniform_location : enable"
			//<< "#extension GL_ARB_shader_image_load_store : enable"
			<< "in vec4 ciPosition;"
			<< "in vec2 ciTexCoord0;"
			<< "out highp vec2 tc;"
			<< "out highp vec2 relOutTc;" // relative out texcoord
			<< "uniform vec2 uTexCoordOffset, uTexCoordScale;"
			<< uniformDeclarations

			<< "void main()"
			<< "{"
			<< "	gl_Position = ciPosition * 2 - 1;"
			<< "	tc = ciTexCoord0;"
			<< "	relOutTc = tc;"
			<< "	tc = uTexCoordOffset + uTexCoordScale * tc;"
			<< opts._vshaderExtra
			<< "}";
			try{
			auto fmt = gl::GlslProg::Format()
				.vertex(completeVshader)
				.fragment(completeFshader)
				.attribLocation("ciPosition", 0)
				.attribLocation("ciTexCoord0", 1)
				.preprocess(false)
				;
			shader = gl::GlslProg::create(fmt);
			shaders[fshader] = shader;
		} catch(gl::GlslProgCompileExc const& e) {
			cout << "gl::GlslProgCompileExc: " << e.what() << endl;
			cout << "source:" << endl;
			cout << completeFshader << endl;
			string s; cin >> s;
			throw;
		}
	} else {
		shader = shaders[fshader];
	}
	auto tex0 = texv[0];
	auto prevGlslProg = gl::Context::getCurrent()->getGlslProg();
	//glUseProgram(shader->getHandle());
	//gl::Context::getCurrent()->bindGlslProg(shader);
	if (shader->getHandle() == 0) {
		cout << "hey" << endl;
	}
	shader->bind();
	//glUseProgram(shader->getHandle());
	ivec2 viewportSize(
		floor(tex0->getWidth() * opts._scaleX),
		floor(tex0->getHeight() * opts._scaleY)
	);

	if (opts._dstRectSize != ivec2()) {
		viewportSize = opts._dstRectSize;
	}
	vector<gl::TextureRef> results;
	if (opts._enableResult) {
		if (opts._targetTexs.size() != 0)
		{
			results = opts._targetTexs;
		}
		else {
			GLenum ifmt = opts._ifmt.exists ? opts._ifmt.val : tex0->getInternalFormat();
			results = { maketex(viewportSize.x, viewportSize.y, ifmt) };
		}
	}
	
	int location = 0;
	shader->uniform("viewportSize", viewportSize);
	location++;
	shader->uniform("mouse", vec2(mouseX, mouseY));
	//::setUniform(location++, vec2(result->getSize()));
	shader->uniform("tex", 0); bindTexture(tex0, GL_TEXTURE0 + 0);
	shader->uniform("texSize", vec2(tex0->getSize()));
	shader->uniform("tsize", vec2(1.0) / vec2(tex0->getSize()));
	for (int i = 1; i < texv.size(); i++) {
		//shader.
		//string index = texIndex(texv[i]);
		shader->uniform(samplerName(i), i); bindTexture(texv[i], GL_TEXTURE0 + i);
		shader->uniform(samplerName(i)+"Size", vec2(texv[i]->getSize()));
		shader->uniform("tsize" + samplerSuffix(i), vec2(1)/vec2(texv[i]->getSize()));
	}
	for (auto& uniform : opts._uniforms)
	{
		uniform.setter(shader);
	}
	/*shader->uniform("image", 0); // todo: rm this?
	if (opts._targetImg != nullptr) {
		my_assert(opts._targetImg->getInternalFormat() == GL_R32F);
		glBindImageTexture(0, opts._targetImg->getId(), 0, GL_FALSE, 0, GL_READ_WRITE, opts._targetImg->getInternalFormat());
	}*/
	auto srcArea = opts._area;
	if (srcArea == Area::zero()) {
		srcArea = tex0->getBounds();
	}
	tex0->setTopDown(true);
	Rectf texRect = tex0->getAreaTexCoords(srcArea);
	tex0->setTopDown(false);
	shader->uniform("uTexCoordOffset", texRect.getUpperLeft());
	shader->uniform("uTexCoordScale", texRect.getSize());

	//glUseProgram(shader->getHandle()); // we did this further up, but the ->uniform calls have messed it up
	shader->bind();

	if (opts._enableResult) {
		beginRTT(results);
	}
	else {
		// if we don't do that, OpenGL clamps the viewport (that we set) by the cinder window size
		beginRTT(opts._targetImg);

		glColorMask(false, false, false, false);
	}
	 
	gl::pushMatrices();
	{
		gl::ScopedViewport sv(opts._dstPos, viewportSize);
		gl::setMatricesWindow(ivec2(1, 1), true);
		gl::disableAlphaBlending();
		::drawRect();
		gl::popMatrices();
	}
	if (opts._enableResult) {
		endRTT();
	}
	else {
		endRTT();
		glColorMask(true, true, true, true);
	}
	//glUseProgram(0); // as in gl::Context::pushGlslProg
	//gl::Context::getCurrent()->bindGlslProg(prevGlslProg);
	//glUseProgram(prevGlslProg->getHandle());
	prevGlslProg->bind();
	return results[0];
}

inline gl::TextureRef shade(vector<gl::TextureRef> const & texv, std::string const & fshader, float resScale)
{
	return shade(texv, fshader, ShadeOpts().scale(resScale));
}

GpuScope::GpuScope(string name) {
	glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, name.c_str());
}

GpuScope::~GpuScope() {
	glPopDebugGroup();
}

ShadeOpts::ShadeOpts() {
	//_ifmt=GL_RGBA16F;
	_scaleX = _scaleY = 1.0f;
}
