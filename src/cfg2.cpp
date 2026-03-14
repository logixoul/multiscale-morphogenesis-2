#include "precompiled.h"
#include "cfg2.h"
#include "CinderImGui.h"


cfg2::cfg2()
{
	ImGui::Initialize();
	ImGuiIO& io = ImGui::GetIO();
	io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);

	tbl = toml::parse_file("config.toml");
}

void cfg2::begin()
{
	ImGui::Begin("Parameters");
}

void cfg2::end()
{
	ImGui::End();
}

template<class T> T& getOpt_Base(string const& name, T defaultValue) {
	static map<string, T> m;
	if (!m.count(name)) {
		m[name] = defaultValue;
	}
	return m[name];
}

/*int cfg2::getInt(string const& name, int min, int max, int defaultValue, ImGuiSliderFlags flags) {

	auto& ref = getOpt_Base<int>(name, defaultValue);
	ImGui::DragInt(name.c_str(), &ref, 1.0f, min, max, "%d", flags);
	return ref;
}*/

// Note: using value_or throughout to handle the "entire table doesn't exist" possibility
float cfg2::getFloat(string const& name) {
	auto subTable = tbl.at_path("param." + name);
	float& ref = getOpt_Base<float>(name, subTable["default"].value_or(0.5));

	ImGui::DragFloat(
		name.c_str(),
		&ref,
		subTable["speed"].value_or(.1),
		subTable["min"].value_or(-100.0),
		subTable["max"].value_or(100.0),
		"%.3f",
		subTable["logarithmic"].value_or(false) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);

	return ref;
}

// Note: using value_or throughout to handle the "entire table doesn't exist" possibility
bool cfg2::getBool(string const& name) {
	auto val = tbl.at_path("param." + name);
	auto& ref = getOpt_Base<bool>(name, val["default"].value_or(false));
	ImGui::Checkbox(name.c_str(), &ref);
	return ref;
}
