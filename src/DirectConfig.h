#pragma once
#include "SimpleGUI/SimpleGUI.h"

class DirectConfig
{
public:
	static float getFloat(string name, float min, float max, float defaultValue);

	static void setSimpleGui(mowa::sgui::SimpleGUI* sgui);

	static void setChangeHandler(string name, function<void()> onValueChanged); // to be called from mainscenewidget

private:
	struct Option {
		float value;
		function<void()> onValueChanged = []() {};
		mowa::sgui::FloatVarControl* control = nullptr;
		bool gottenIt = false; // todo improve
	};

	DirectConfig();

	static mowa::sgui::SimpleGUI* m_sgui;

	static std::map<string, Option> floats;
};
