#pragma once

#include "precompiled.h"

// base class providing the "event accepting" functionality of event propagation
	// (see "Accept or ignore" at http://doc.qt.nokia.com/qq/qq11-events.html, we're doing things the same way)
	//
	// IMPORTANT! The "Accepted" functionality is not yet used anywhere.
class CGPropagatedEventArgs
{
public: bool Accepted;

public: CGPropagatedEventArgs()
{
	Accepted = true;
}

public: void Accept()
{
	Accepted = true;
}

public: void Ignore()
{
	Accepted = false;
}
};

// This class is like OpenTK's MouseEventArgs, but it inherits CGPropagatedEventArgs.
class CGMouseEventArgs : public CGPropagatedEventArgs
{
public: CGMouseEventArgs(ivec2 position)
{
	Position = position;
}

public: ivec2 Position;
};

enum class CGMouseButton {
	Left,
	Middle,
	Right
};

class CGMouseButtonEventArgs : public CGMouseEventArgs
{
public: CGMouseButton Button;
public: bool IsPressed;

public: CGMouseButtonEventArgs(ivec2 position, CGMouseButton button, bool isPressed)
	: CGMouseEventArgs(position)
	{
		IsPressed = isPressed;
		Button = button;
	}

	CGMouseButtonEventArgs(ci::app::MouseEvent e) : CGMouseEventArgs(e.getPos()) {
		if (
			(e.isLeft() && e.isLeftDown())
			|| (e.isRight() && e.isRightDown())
			|| (e.isMiddle() && e.isMiddleDown())
			)
			IsPressed = true;
		if (e.isLeft()) Button = CGMouseButton::Left;
		else if (e.isRight()) Button = CGMouseButton::Right;
		else if (e.isMiddle()) Button = CGMouseButton::Middle;
		else throw 0;
	}

	operator ci::app::MouseEvent() {
		int initiator = 0;
		if (Button == CGMouseButton::Left) initiator = ci::app::MouseEvent::LEFT_DOWN;
		if (Button == CGMouseButton::Right) initiator = ci::app::MouseEvent::RIGHT_DOWN;
		if (Button == CGMouseButton::Middle) initiator = ci::app::MouseEvent::MIDDLE_DOWN;
		ci::app::MouseEvent res(ci::app::getWindow(), initiator, Position.x, Position.y, 0, 0, 0);
		return res;
	}
};

class CGMouseMoveEventArgs : public CGMouseEventArgs
{
public: CGMouseMoveEventArgs(ivec2 position, ivec2 delta)
	: CGMouseEventArgs(position)
{
	Delta = delta;
}

public: ivec2 Delta;

		ivec2 previousPosition() {
			return Position - Delta;
		}

		/*CGMouseMoveEventArgs(ci::app::MouseEvent e) : CGMouseEventArgs(e.getPos()) {
			Delta = 
		}*/

		operator ci::app::MouseEvent() {
			ci::app::MouseEvent res(ci::app::getWindow(), 0, Position.x, Position.y, 0, 0, 0);
			return res;
		}
};

class CGMouseWheelEventArgs : public CGMouseEventArgs
{
public: int Delta;
public: float DeltaPrecise;
public: int Value;
public: float ValuePrecise;

public: CGMouseWheelEventArgs(ivec2 position, int delta, float deltaPrecise, int value, float valuePrecise)
	: CGMouseEventArgs(position)
{
	ValuePrecise = valuePrecise;
	Value = value;
	DeltaPrecise = deltaPrecise;
	Delta = delta;
}

		operator ci::app::MouseEvent() {
			ci::app::MouseEvent res(ci::app::getWindow(), 0, Position.x, Position.y, DeltaPrecise, 0, 0);
			return res;
		}
};

class CGKeyboardKeyEventArgs : public CGPropagatedEventArgs
{
public: char Key;

public: CGKeyboardKeyEventArgs(char key)
{
	Key = key;
}
};

class FocusEventArgs : public CGPropagatedEventArgs
{
public:
	// specifies whether the widget/window got focus or lost focus.
	bool GotFocus;

	FocusEventArgs(bool gotFocus)
	{
		GotFocus = gotFocus;
	}
};

class CGCancelEventArgs
{
public:
	bool Cancel;

	CGCancelEventArgs(bool cancel)
	{
		Cancel = cancel;
	}
};