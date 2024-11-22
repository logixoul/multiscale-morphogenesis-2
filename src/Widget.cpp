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
#include "Widget.h"

Widget::Widget(Widget * parent) { m_parent = parent; }

float Widget::wscale() { return getWindowContentScale() * 2; }

Widget::~Widget() {
	for (auto child : m_children)
	{
		delete child;
	}
	// todo: remove self from m_parent->m_children
}

// If this widget is a subwindow, it gets closed. Otherwise, an exception is thrown.

void Widget::close()
{
	throw 0;
	/*bool canceled;
	GuiManager::instance().closeSubwindow(this, &canceled);
	if (!canceled)
		Dispose();*/
}

ivec2 Widget::size() {
	return m_size;
}

void Widget::setSize(ivec2 val) {
	m_size = val;
}

void Widget::setLocation(ivec2 val) {
	m_location = val;
}

ivec2 Widget::location() {
	return m_location;
}

Area Widget::geometry() {
	return Area(m_location, m_size);
}

void Widget::setGeometry(Area val) {
	m_location = val.getUL();
	m_size = val.getSize();
}

ivec2 Widget::absoluteLocation()
{
	if (m_parent == nullptr)
		return m_location;
	else
		return m_location + m_parent->absoluteLocation();
}

Area Widget::absoluteGeometry() { return Area(absoluteLocation(), m_size); }

vector<Widget*>& Widget::children() {
	return m_children;
}

void Widget::removeChildLater(Widget * child)
{
	m_childrenToRemove.push_back(child);
}

bool Widget::visible() {
	return m_visible;
}

void Widget::setVisible(bool val) {
	m_visible = val;
}

bool Widget::enabled() {
	return m_enabled;
}

void Widget::setEnabled(bool val) {
	m_enabled = val;
}

bool Widget::enabledUpToRoot()
{
	if (m_parent == nullptr)
		return m_enabled;
	else
		return m_enabled && m_parent->enabledUpToRoot();
}

Widget * Widget::parent() {
	return m_parent;
}

Widget * Widget::focusedWidget() {
	return m_focusedWidget;
}

void Widget::setFocusedWidget(Widget * val) {
	if (m_focusedWidget != nullptr)
		m_focusedWidget->OnWidgetFocusedChanged(FocusEventArgs(false));

	m_focusedWidget = val;

	if (m_focusedWidget != nullptr)
		m_focusedWidget->OnWidgetFocusedChanged(FocusEventArgs(true));
}

template<class TElem>
static vector<TElem> setDifference(vector<TElem> const& a, vector<TElem> const& toRemove) {
	vector<TElem> aCopy = a;
	for (TElem const& elem : toRemove) {
		// https://stackoverflow.com/a/39944
		aCopy.erase(std::remove(aCopy.begin(), aCopy.end(), elem), aCopy.end());
	}
	return aCopy;
}

void Widget::removeScheduledChildren()
{
	m_children = ::setDifference(m_children, m_childrenToRemove);
	m_childrenToRemove.clear();;
}
