#include "precompiled.h"
#include "EventDispatcher.h"
#include "Widget.h"
#include "GuiManager.h"
#include "System.h"

EventDispatcher::EventDispatcher(GuiManager * guiManager)
{
	m_guiManager = guiManager;
}

CGMouseButtonEventArgs EventDispatcher::MapToLocalCoordSystem(CGMouseButtonEventArgs e, Widget * receiver)
{
	auto localPos = e.Position - receiver->absoluteLocation();
	return CGMouseButtonEventArgs(localPos, e.Button, e.IsPressed);
}

CGMouseMoveEventArgs EventDispatcher::MapToLocalCoordSystem(CGMouseMoveEventArgs e, Widget * receiver)
{
	auto localPos = e.Position - receiver->absoluteLocation();
	return CGMouseMoveEventArgs(localPos, e.Delta);
}

CGMouseWheelEventArgs EventDispatcher::MapToLocalCoordSystem(CGMouseWheelEventArgs e, Widget * receiver)
{
	auto localPos = e.Position - receiver->absoluteLocation();
	return CGMouseWheelEventArgs(localPos, e.Delta, e.DeltaPrecise, e.Value, e.ValuePrecise);
}

bool EventDispatcher::WantsEvents(Widget * receiver)
{
	return receiver->visible() && /*!receiver.IgnoreEvents &&*/ receiver->enabledUpToRoot();
}

Widget * EventDispatcher::GetTopmostWidgetConcernedByMouseEvent(Widget * root, ivec2 eventLocation)
{
	if (!WantsEvents(root) || !root->absoluteGeometry().contains(eventLocation))
		return nullptr;
	auto reversedChildren = root->children();
	std::reverse(reversedChildren.begin(), reversedChildren.end());
	for (auto child : reversedChildren)
	{
		Widget* fromChild = GetTopmostWidgetConcernedByMouseEvent(child, eventLocation);
		if (fromChild != nullptr)
			return fromChild;
	}
	return root;
}

Widget * EventDispatcher::InvokeUntilAccepted(Widget * widget, std::function<bool(Widget*)> const & ev)
{
	if (WantsEvents(widget) && ev(widget))
		return widget;
	else if (widget->parent() != nullptr)
	{
		return InvokeUntilAccepted(widget->parent(), ev);
	}

	return nullptr;
}

void EventDispatcher::OnMouseButtonDown(CGMouseButtonEventArgs e)
{
	Widget* concernedWidget = GetTopmostWidgetConcernedByMouseEvent(m_guiManager->topSubwindow(), e.Position);
	if (concernedWidget != nullptr)
	{
		// the widgets notified by InvokeUntilAccepted may change the TopSubwindow (e.g. by popping up a new one). We still want
		// to only modify the focus of the subwindow that was the top one *before* the invokes.
		Widget* topSubwindowBeforeInvokes = m_guiManager->topSubwindow();
		auto invoked = InvokeUntilAccepted(concernedWidget, [&](Widget* widget) { widget->onMouseDown(MapToLocalCoordSystem(e, widget)); return e.Accepted; });
		if (invoked != nullptr)
		{
			m_mouseGrabber = invoked;
		}

		topSubwindowBeforeInvokes->setFocusedWidget(invoked);
	}
}

void EventDispatcher::OnMouseButtonUp(CGMouseButtonEventArgs e)
{
	if (m_mouseGrabber == nullptr)
	{
		Widget* concernedWidget = GetTopmostWidgetConcernedByMouseEvent(m_guiManager->topSubwindow(), e.Position);
		if (concernedWidget != nullptr)
		{
			m_mouseGrabber = concernedWidget;
			InvokeUntilAccepted(concernedWidget, [&](Widget* widget) { widget->onMouseUp(MapToLocalCoordSystem(e, widget)); return e.Accepted; });
		}
	}
	else
	{
		m_mouseGrabber->onMouseUp(MapToLocalCoordSystem(e, m_mouseGrabber));
		m_mouseGrabber = nullptr;
	}
}

void EventDispatcher::OnMouseMotion(CGMouseMoveEventArgs e)
{
	const bool dragging = System::isMouseButtonHeld();
	function<void(Widget*, CGMouseMoveEventArgs ev)> moveHandlerToCall =
		[&](Widget* w, CGMouseMoveEventArgs ev) { dragging ? w->onMouseDrag(ev) : w->onMouseMove(ev); }
	;
	if (m_mouseGrabber == nullptr)
	{
		// we care both about the prev pos and the current pos.
		Widget* widgetConcernedByPrevPos = GetTopmostWidgetConcernedByMouseEvent(m_guiManager->topSubwindow(), e.previousPosition());
		Widget* widgetConcernedByCurPos = GetTopmostWidgetConcernedByMouseEvent(m_guiManager->topSubwindow(), e.Position);
		if (widgetConcernedByCurPos != widgetConcernedByPrevPos)
		{
			if (widgetConcernedByPrevPos != nullptr)
				InvokeUntilAccepted(widgetConcernedByPrevPos, [&](Widget* widget) {
					widget->OnMouseLeave(); return e.Accepted; });
			if (widgetConcernedByCurPos != nullptr)
				InvokeUntilAccepted(widgetConcernedByCurPos, [&](Widget* widget) {
					widget->OnMouseEnter(); return e.Accepted; });
		}
		// we notify both.
		if (widgetConcernedByPrevPos != nullptr)
		{
			InvokeUntilAccepted(widgetConcernedByPrevPos, [&](Widget* widget) {
				moveHandlerToCall(widget, MapToLocalCoordSystem(e, widget)); return e.Accepted; });
		}
		if (widgetConcernedByCurPos != nullptr)
		{
			InvokeUntilAccepted(widgetConcernedByCurPos, [&](Widget* widget) {
				moveHandlerToCall(widget, MapToLocalCoordSystem(e, widget)); return e.Accepted; });
		}
	}
	else
	{
		m_mouseGrabber->onMouseDrag(MapToLocalCoordSystem(e, m_mouseGrabber));
	}
}

void EventDispatcher::OnKeyDown(CGKeyboardKeyEventArgs e)
{
	auto window = m_guiManager->topSubwindow();
	InvokeUntilAccepted(window->focusedWidget() ? window->focusedWidget() : window, [&](Widget* widget) {  widget->onKeyDown(e); return e.Accepted; });
}

void EventDispatcher::OnKeyUp(CGKeyboardKeyEventArgs e)
{
	auto window = m_guiManager->topSubwindow();
	InvokeUntilAccepted(window->focusedWidget() ? window->focusedWidget() : window, [&](Widget* widget) { widget->onKeyUp(e); return e.Accepted; });
}

void EventDispatcher::OnMouseWheelMoved(CGMouseWheelEventArgs e)
{
	Widget* concernedWidget = GetTopmostWidgetConcernedByMouseEvent(m_guiManager->topSubwindow(), e.Position);
	if (concernedWidget != nullptr)
	{
		InvokeUntilAccepted(concernedWidget, [&](Widget* widget) {  widget->OnMouseWheelMoved(MapToLocalCoordSystem(e, widget)); return e.Accepted; });
	}
}

void EventDispatcher::OnWindowFocusedChanged(FocusEventArgs e)
{
	Widget* receiver = m_guiManager->topSubwindow();
	if (!WantsEvents(receiver))
		return;
	receiver->OnWindowFocusedChanged(e);
}

void EventDispatcher::OnWindowClosing(CGCancelEventArgs e)
{
	m_guiManager->topSubwindow()->OnSubwindowClosing(e);
	if (e.Cancel)
		return;

	Widget* receiver = m_guiManager->topSubwindow();
	if (!WantsEvents(receiver))
	{
		e.Cancel = true;
		return;
	}
	receiver->OnWindowClosing(e);
}

void EventDispatcher::OnWindowClosed()
{
	Widget* receiver = m_guiManager->topSubwindow();
	if (!WantsEvents(receiver))
		return;
	receiver->OnWindowClosed();
}

void EventDispatcher::onFileDrop(FileDropEvent ev)
{
	Widget* receiver = m_guiManager->topSubwindow();
	if (!WantsEvents(receiver))
		return;
	receiver->onFileDrop(ev);
}
