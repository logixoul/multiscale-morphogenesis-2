#include "precompiled.h"
#include "GuiManager.h"
#include "Widget.h"
#include "events.h"
#include "MyTimer.h"

GuiManager::GuiManager()
	: m_eventDispatcher(this)
{
}

// todo: make GuiManager not be a singleton
GuiManager::~GuiManager()
{
	for (auto subwindow : m_subwindowStack) {
		delete subwindow;
	}
}

GuiManager& GuiManager::instance()
{
	static GuiManager guiManager;
	return guiManager;
}

void GuiManager::render()
{
	gl::pushMatrices(); // todo: just model matrix?
	gl::setMatricesWindow(ci::app::getWindowSize());

	if (m_renderOnlyTopSubwindow) {
		renderWidgetTree(topSubwindow());
	}
	else {
		// make a copy so the loop doesn't fail if the collection is changed in the meantime
		vector<Widget*> subwindowStackCopy = m_subwindowStack;
		for(auto subwindow : subwindowStackCopy)
		{
			renderWidgetTree(subwindow);
		}
	}

	gl::popMatrices();
}

void GuiManager::update()
{
	// make a copy so the loop doesn't fail if the collection is changed in the meantime
	vector<Widget*> subwindowStackCopy = m_subwindowStack;
	for (auto subwindow : subwindowStackCopy)
	{
		updateWidgetTree(subwindow);
	}
}

void GuiManager::addSubwindow(Widget * newSubwindow)
{
	m_subwindowStack.push_back(newSubwindow);
}

void GuiManager::renderWidgetTree(Widget* root)
{
	if (!root->visible()) return;


	gl::pushMatrices(); // todo
	gl::translate(root->absoluteLocation());
	root->draw();
	gl::popMatrices();

	for(auto child : root->children())
	{
		renderWidgetTree(child);
	}
}

void GuiManager::updateWidgetTree(Widget * root)
{
	//==========
	//cout << "BAD HACK in updateWidgetTree" << endl;
	root->setSize(getWindowSize());
	//==========



	root->removeScheduledChildren();

	root->update();

	root->layoutChildren();
	for (auto child : root->children())
	{
		updateWidgetTree(child);
	}
}

void GuiManager::closeSubwindow(Widget* subwindow, bool * canceled)
{
	auto it = std::find(m_subwindowStack.begin(), m_subwindowStack.end(), subwindow);
	if (it == m_subwindowStack.end())
		throw new Exception("Widget is not a subwindow");

	auto eventArgs = CGCancelEventArgs(false);
	subwindow->OnSubwindowClosing(eventArgs);
	bool dummy;
	if (canceled == nullptr)
		canceled = &dummy;
	*canceled = eventArgs.Cancel;
	if (*canceled)
		return;
	subwindow->OnSubwindowClosed();

	// remove this subwindow, as well all subwindow after it (because they were created by it)
	m_subwindowStack.erase(it, m_subwindowStack.end());
	MyTimer::singleShot(0, [subwindow]() {
		delete subwindow;
	});
}

Widget * GuiManager::topSubwindow() {
	return m_subwindowStack[m_subwindowStack.size() - 1];
}
