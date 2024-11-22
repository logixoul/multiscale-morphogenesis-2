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
#include "MessageWidget.h"
#include "SimpleGUI/SimpleGUI.h"

using namespace mowa::sgui;

MessageWidget::MessageWidget(ButtonType bType)
	: Widget(0) // todo
{
	auto app = (App*)App::get();
	gui = new SimpleGUI(app);

	if (bType == ButtonType::Ok) {
		button = gui->addButton("OK");
	}
}

void MessageWidget::draw()
{
	computeLayout();
	gl::pushMatrices();
	gl::setMatricesWindow(getWindowSize(), true);
	glDisable(GL_DEPTH_TEST);
	gl::ScopedBlendAlpha sba;
	{
		gl::ScopedColor sc(0, 0, 0, .7f);
		auto glsl = gl::getStockShader(gl::ShaderDef().color());
		gl::ScopedGlslProg glslScp(glsl);
		//gl::color(ColorA(0, 0, 0, .5));
		gl::drawSolidRect(getWindowBounds());
	}
	gl::drawStringCentered(text, getWindowCenter(), ColorA(1, 1, 1, .5), SimpleGUI::textFont);
	gl::popMatrices();

	gui->draw();
}

void MessageWidget::onMouseDown(CGMouseButtonEventArgs ev)
{
	gui->onMouseDown(ev);
}

void MessageWidget::onMouseUp(CGMouseButtonEventArgs ev)
{
	gui->onMouseUp(ev);
}

void MessageWidget::onMouseDrag(CGMouseMoveEventArgs ev)
{
	gui->onMouseDrag(ev);
}

void MessageWidget::onMouseMove(CGMouseMoveEventArgs ev)
{
	gui->onMouseMove(ev);
}

void MessageWidget::computeLayout()
{
	if (button != nullptr) {
		auto center = getWindowCenter() + vec2(0, 90 * wscale());
		auto rect = Rectf(center, center);
		rect.inflate(vec2(50, 15) * wscale());
		button->optionalGeom = rect;
	}
}

