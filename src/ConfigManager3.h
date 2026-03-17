#pragma once
#include "precompiled.h"
#include "CinderImGui.h"
#include "toml.hpp"

struct ConfigManager3
{
private:
	toml::table tbl;
public:
	ConfigManager3();
	bool getBool(string const& name);
	//int getInt(string const& name, int min, int max, int defaultValue, ImGuiSliderFlags flags = ImGuiSliderFlags_::ImGuiSliderFlags_None);
	float getFloat(string const& name);
	void begin();
	void end();
};
