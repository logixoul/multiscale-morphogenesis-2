#pragma once

#include "EventDispatcher.h"

class Widget;

class GuiManager
{
public:
	GuiManager();
	~GuiManager();

	static GuiManager& instance();
	void render();
	void update();

	void addSubwindow(Widget* newSubwindow);
	void closeSubwindow(Widget* subwindow, bool* canceled = nullptr);

	Widget* topSubwindow();

	EventDispatcher& eventDispatcher() {
		return m_eventDispatcher;
	}

private:
	EventDispatcher m_eventDispatcher;
	vector<Widget*> m_subwindowStack;
	bool m_renderOnlyTopSubwindow;

	void renderWidgetTree(Widget* root);
	void updateWidgetTree(Widget* root);
};

