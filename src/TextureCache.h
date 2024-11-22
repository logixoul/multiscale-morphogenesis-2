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

struct TextureCacheKey {
	ivec2 size;
	GLenum ifmt;
	bool allocateMipmaps = false;
	
	bool operator==(const TextureCacheKey &other) const
	{
		return size == other.size
			&& ifmt == other.ifmt
			&& allocateMipmaps == other.allocateMipmaps
			;
	}
};

namespace std {

	template <>
	struct hash<TextureCacheKey>
	{
		std::size_t operator()(const TextureCacheKey& k) const
		{
			return k.size.x ^ k.size.y ^ k.ifmt ^ k.allocateMipmaps;
		}
	};

}

class TextureCache
{
public:
	static TextureCache* instance();
	gl::TextureRef get(TextureCacheKey const& key);
	static void clearCaches();

	static void printTextures();

	static void deleteTexture(Tex tex);

private:
	TextureCache();

	std::unordered_map<TextureCacheKey, vector<gl::TextureRef>> cache;
};
