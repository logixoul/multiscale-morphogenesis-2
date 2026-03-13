#pragma once
#include "precompiled.h"
#include "CinderImGui.h"
#include "toml.hpp"

struct cfg2
{
private:
	toml::table tbl;
public:
	cfg2();
	bool getBool(string const& name);
	int getInt(string const& name, int min, int max, int defaultValue, ImGuiSliderFlags flags = ImGuiSliderFlags_::ImGuiSliderFlags_None);
	float getFloat(string const& name);
	void begin();
	void end();
};
