#include "precompiled.h"
#include "DirectConfig.h"

mowa::sgui::SimpleGUI* DirectConfig::m_sgui = nullptr;
std::map<string, DirectConfig::Option> DirectConfig::floats;

float DirectConfig::getFloat(string name, float min, float max, float defaultValue)
{
	if (!floats[name].gottenIt) {
		//floats.insert(make_pair(name, Option()));
		floats[name].control = m_sgui->addParam(name, &floats[name].value, min, max, defaultValue);
		floats[name].control->valueChanged.connect(floats[name].onValueChanged);
		floats[name].gottenIt = true;
	}
	return floats[name].value;
}

void DirectConfig::setSimpleGui(mowa::sgui::SimpleGUI * sgui)
{
	m_sgui = sgui;
}

void DirectConfig::setChangeHandler(string name, function<void()> onValueChanged)
{
	//if(floats[name].control != nullptr)
	floats[name].onValueChanged = onValueChanged;
}
