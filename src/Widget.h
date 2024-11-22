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

#include "events.h"

class Widget
{
public:
	explicit Widget(Widget* parent);

	virtual void update() { }
	virtual void layoutChildren() { }
	virtual void draw() { }
	virtual void onFileDrop(FileDropEvent ev) { }
	virtual void onMouseDown(CGMouseButtonEventArgs ev) { }
	virtual void onMouseUp(CGMouseButtonEventArgs ev) { }
	virtual void onKeyDown(CGKeyboardKeyEventArgs ev) { }
	virtual void onKeyUp(CGKeyboardKeyEventArgs ev) { }
	virtual void onMouseDrag(CGMouseMoveEventArgs ev) { }
	virtual void onMouseMove(CGMouseMoveEventArgs ev) { }
	virtual void OnWidgetFocusedChanged(FocusEventArgs ev) { }
	virtual void OnMouseLeave() { }
	virtual void OnMouseEnter() { }
	virtual void OnMouseWheelMoved(CGMouseWheelEventArgs ev) { }
	virtual void OnWindowFocusedChanged(FocusEventArgs ev) {}
	virtual void OnSubwindowClosing(CGCancelEventArgs ev) { }
	virtual void OnSubwindowClosed() { }
	virtual void OnWindowClosing(CGCancelEventArgs ev) { }
	virtual void OnWindowClosed() { }

	float wscale();

	virtual ~Widget();

	// If this widget is a subwindow, it gets closed. Otherwise, an exception is thrown.
	virtual void close();

	// ================ GEOMETRY STUFF

	ivec2 size();

	void setSize(ivec2 val);

	void setLocation(ivec2 val);

	ivec2 location();

	Area geometry();

	void setGeometry(Area val);

	ivec2 absoluteLocation();

	Area absoluteGeometry();

	// ================ TREE STUFF

	vector<Widget*>& children();

	void removeChildLater(Widget* child);

	bool visible();

	void setVisible(bool val);

	bool enabled();

	void setEnabled(bool val);

	bool enabledUpToRoot();

	Widget* parent();

	Widget* focusedWidget();

	void setFocusedWidget(Widget* val);

private:
	vector<Widget*> m_children;
	vector<Widget*> m_childrenToRemove;
	Widget* m_parent;
	ivec2 m_size;
	ivec2 m_location;
	bool m_visible = true;
	bool m_enabled = true;
	bool m_currentlyManipulated = false;
	// note: only root widgets have a FocusedWidget.
	Widget* m_focusedWidget = nullptr;


	void removeScheduledChildren();
	friend class GuiManager;
};

