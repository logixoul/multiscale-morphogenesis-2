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
#include "precompiled.h"

extern float mouseX, mouseY;
extern bool keys[256];
extern bool keys2[256];
extern bool mouseDown_[3];
extern int wsx, wsy; // define and initialize those in main.cpp

// stefan's framework
namespace stefanfw {
	void beginFrame();
	void endFrame();
	struct EventHandler {
		bool keyDown(KeyEvent e);
		bool keyUp(KeyEvent e);
		bool mouseDown(MouseEvent e);
		bool mouseUp(MouseEvent e);
		void subscribeToEvents(ci::app::App& app);
	} extern eventHandler;
};