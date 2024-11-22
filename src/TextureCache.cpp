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
#include "TextureCache.h"
#include "stuff.h"

int count1;
int count2;

TextureCache* TextureCache::instance() {
	static /*thread_local*/ TextureCache obj;
	return &obj;
}

std::map<int, string> fmtMap = {
	{ GL_R16F, "GL_R16F" },
	{ GL_RGB16F, "GL_RGB16F" },
	{ GL_RG32F, "GL_RG32F" },
	{ GL_R32F, "GL_R32F" },
	{ GL_RGBA16F, "GL_RGBA16F" },
	{ GL_RGB8, "GL_RGB8" },
};

std::map<int, int> fmtMapBpp = {
	{ GL_R16F, 2 },
	{ GL_R32F, 4 },
	{ GL_RGB16F, 6 },
	{ GL_RG32F, 8 },
	{ GL_RGBA16F, 8 },
	{ GL_RGB8, 3 },
};

TextureCache::TextureCache() {
}

static gl::TextureRef allocTex(TextureCacheKey const& key) {
	//cout << "allocTex\t" << key.size << "\t" << fmtMap[key.ifmt] << endl;
	gl::Texture::Format fmt;
	fmt.setInternalFormat(key.ifmt);
	fmt.setImmutableStorage(true);
	fmt.enableMipmapping(key.allocateMipmaps);
	auto tex = gl::Texture::create(key.size.x, key.size.y, fmt);
	return tex;
}

void setDefaults(gl::TextureRef tex) {
	tex->setMinFilter(GL_LINEAR);
	tex->setMagFilter(GL_LINEAR);
	tex->setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
}

//static void clearContents(gl::TextureRef const& tex) {
//}

static std::mutex mut;

gl::TextureRef TextureCache::get(TextureCacheKey const & key)
{
	unique_lock<std::mutex> ul(::mut);
	/*auto t = allocTex(key);
	setDefaults(t);
	return t;*/

	auto it = cache.find(key);
	if (it == cache.end()) {
		auto tex = allocTex(key);
		vector<gl::TextureRef> vec{ tex };
		cache.insert(std::make_pair(key, vec));
		setDefaults(tex);
		return tex;
	}
	else {
		auto& vec = it->second;
		for (auto& texRef : vec) {
			if (texRef.use_count() == 1) {
				count2++;
				//cout << "returning existing texture" << endl;
				setDefaults(texRef);
				return texRef;
			}
		}
		//cout << "requesting utilized texture; returning new texture." << endl;
		count1++;
		if (key.ifmt == GL_R32F && key.size.x == 5312) {
			cout << "hey" << endl;
		}

		auto tex = allocTex(key);
		vec.push_back(tex);
		setDefaults(tex);
		return tex;
	}
}

// todo: rename this func
void TextureCache::clearCaches()
{
	unique_lock<std::mutex> ul(::mut);
	auto& cache = instance()->cache;
	
	for (auto& pair : cache) {
		vector<gl::TextureRef> remaining;
		for (auto& tex : pair.second) {
			if (tex.use_count() > 1) {
				remaining.push_back(tex);
				//cout << "retaining " << tex->getSize() << endl;
			}
			else {
				//cout << "deleting " << tex->getSize() << endl;
			}
		}
		//pair.second = remaining;
		cache[pair.first] = remaining;
	}
}

void TextureCache::printTextures()
{
	unique_lock<std::mutex> ul(::mut);
	cout << "==============" << endl;
	int totalBytes = 0;
	std::map<gl::TextureRef, bool> hasMips;
	auto vec = vector<gl::TextureRef>();
	for (auto& pair : instance()->cache) {
		for (auto& tex : pair.second) {
			vec.push_back(tex);
			hasMips[tex] = pair.first.allocateMipmaps;
		}
	}
	sort(vec.begin(), vec.end(), [&](gl::TextureRef a, gl::TextureRef b) {
		auto aBytes = a->getBounds().calcArea() * fmtMapBpp[a->getInternalFormat()];
		auto bBytes = b->getBounds().calcArea() * fmtMapBpp[b->getInternalFormat()];
		return aBytes > bBytes;
	});
	for (auto& tex : vec) {
		auto ifmt = tex->getInternalFormat();
		int bytes = tex->getBounds().calcArea() * fmtMapBpp[ifmt];
		cout << tex->getSize() << " " << fmtMap[ifmt] << "\t" << bytes / 1'000'000 << "MB";
		if(hasMips[tex])
			cout << "\t[has mips]";
		cout << endl;
		totalBytes += bytes;
	}
	cout << "megabytes = " << totalBytes /1'000'000 << endl;
}

void TextureCache::deleteTexture(Tex texToDel)
{
	unique_lock<std::mutex> ul(::mut);
	auto& cache = instance()->cache;

	for (auto& pair : cache) {
		vector<gl::TextureRef> remaining;
		for (auto& tex : pair.second) {
			if (tex != texToDel) {
				remaining.push_back(tex);
				//cout << "retaining " << tex->getSize() << endl;
			}
			else {
				//cout << "deleting " << tex->getSize() << endl;
			}
		}
		//pair.second = remaining;
		cache[pair.first] = remaining;
	}
}
