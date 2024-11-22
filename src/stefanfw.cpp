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
#include "stefanfw.h"
#include "MyTimer.h"
#include "sw.h"

float mouseX, mouseY;
bool keys[256];
bool keys2[256];
bool mouseDown_[3];

namespace stefanfw {


	EventHandler eventHandler;

	void beginFrame() {
		ci::app::AppBase* app = ci::app::App::get();

		sw::beginFrame();

		wsx = app->getWindowWidth();
		wsy = app->getWindowHeight();

		auto relMousePos = app->getMousePos()-app->getWindowPos();
		::mouseX = relMousePos.x / (float)app->getWindow()->getWidth();
		::mouseY = relMousePos.y / (float)app->getWindow()->getHeight();
	}

	void endFrame() {
		//sw::endFrame();
		TimerManager::update();
	}

	// todo make this take ref
	bool EventHandler::keyDown(KeyEvent e) {
		keys[e.getChar()] = true;
		if(e.isControlDown()&&e.getCode()!=KeyEvent::KEY_LCTRL)
		{
			keys2[e.getChar()] = !keys2[e.getChar()];
		}
		return true;
	}

	bool EventHandler::keyUp(KeyEvent e) {
		keys[e.getChar()] = false;
		return true;
	}

	bool EventHandler::mouseDown(MouseEvent e)
	{
		mouseDown_[e.isLeft() ? 0 : e.isMiddle() ? 1 : 2] = true;
		return true;
	}
	bool EventHandler::mouseUp(MouseEvent e)
	{
		mouseDown_[e.isLeft() ? 0 : e.isMiddle() ? 1 : 2] = false;
		return true;
	}

	void EventHandler::subscribeToEvents(ci::app::App& app) {
		auto window = app.getWindow();
		window->getSignalKeyDown().connect([&](KeyEvent& e) { keyDown(e); });
		window->getSignalKeyUp().connect([&](KeyEvent& e) { keyUp(e); });
		window->getSignalMouseDown().connect([&](MouseEvent& e) { mouseDown(e); });
		window->getSignalMouseUp().connect([&](MouseEvent& e) { mouseUp(e); });
	}

} // namespace stefanfw