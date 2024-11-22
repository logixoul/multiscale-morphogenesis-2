#include "precompiled.h"
#include "cfg2.h"
#include "CinderImGui.h"


void cfg2::init()
{
	ImGui::Initialize();
	ImGuiIO& io = ImGui::GetIO();
	io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
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
}

float cfg2::getFloat(string const& name, float min, float max, float defaultValue, ImGuiSliderFlags flags) {
	auto& ref = getOpt_Base<float>(name, defaultValue);
	ImGui::DragFloat(name.c_str(), &ref, min, max, "%d", flags);
	return ref;
}
bool cfg2::getBool(string const& name, bool defaultValue) {
	auto& ref = getOpt_Base<bool>(name, defaultValue);
	ImGui::Checkbox(name.c_str(), &ref);
	return ref;
}
*/