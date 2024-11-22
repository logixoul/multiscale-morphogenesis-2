#pragma once
#include "events.h"
class GuiManager;
class Widget;

// This class is responsible for ensuring that, in a GUI system, the correct widgets become notified of input events.
//
// == Dispatching algorithm ==
// 1. If a mouse event happens, it is sent to the topmost subwindow which further dispatches it:
//    If(there is a mouse grabber)
//    {
//        it receives the event, and dispatching ends.
//    }
//    else
//    {
//        Let x = all widgets containing the mouse position [*].
//        Let y = x except those those widgets that have an ancestor p for which (!p.Visible || p.IgnoreEvents).
//        The event is received by the top-most (in terms of rendering order) widgets among y.
//        Dispatching ends.
//    }
//     [*] Note: in the case of motion events, widgets containing the _previous_ mouse position are also notified. This
//     allows widgets to have "onMouseEnter/onMouseLeave" behavior.
// 2. If a key event happens, it is sent to the topmost subwindow's root widget.
//    Same for window-focus events, window-close events and subwindow-close events.
class EventDispatcher //: IInputListener
{
	private: GuiManager* m_guiManager;

	// The mouseGrabber property is best explained with an example:
	//     1. Mouse is pressed (and held) on slider
	//     2. Mouse is moved far below the slider
	//     3. Mouse is moved horizontally. This should still move the slider even though the mouse is not above it.
	//     4. Ergo, the slider is the mouseGrabber
	// It is null when no widget is capturing events.
	private: Widget* m_mouseGrabber;

public: EventDispatcher(GuiManager* guiManager);

	// DiscardPendingEvents is useful when we do a long load (e.g. between periods in whitecity). During the load the user
	// may do stuff like press space (which normally closes the message), but the app wouldn't react immediately.
	// Instead, the key press event would stay on the OS event queue until the load finishes and we call ProcessEvents.
	// Then we would interpret the pressed key at a bad moment.
	// The fix is to call this function after the load finishes.
	/*public: void DiscardPendingEvents()
	{
		bool wasEnabled = m_guiManager->TopSubwindow.Enabled;
		m_guiManager->TopSubwindow.Enabled = false;
		CGWindow.Instance.InputProvider.ProcessEvents();
		m_guiManager->TopSubwindow.Enabled = wasEnabled;
	}*/

	//================================= IInputListener Members

private: CGMouseButtonEventArgs MapToLocalCoordSystem(CGMouseButtonEventArgs e, Widget* receiver);

private: CGMouseMoveEventArgs MapToLocalCoordSystem(CGMouseMoveEventArgs e, Widget* receiver);

private: CGMouseWheelEventArgs MapToLocalCoordSystem(CGMouseWheelEventArgs e, Widget* receiver);

private: static bool WantsEvents(Widget* receiver);

	// Definition: a widget W is "concerned" by a mouse event if:
	//      it contains the event location and WantsEvents(W) == true.
	//
	// Definition: a widget that is concerned by an event is called "topmost" if no other widget
	// that is concerned by the event is rendered above that leaf.
	// Note: we want "topmost" because of, for example, the case where we have a background ImageWidget.
	//
	// This function returns null when no widget is concerned by the event.
private: Widget* GetTopmostWidgetConcernedByMouseEvent(Widget* root, ivec2 eventLocation);

private: Widget* InvokeUntilAccepted(Widget* widget, std::function<bool(Widget*)> const& ev);

public: void OnMouseButtonDown(CGMouseButtonEventArgs e);

public: void OnMouseButtonUp(CGMouseButtonEventArgs e);

public: void OnMouseMotion(CGMouseMoveEventArgs e);

public: void OnKeyDown(CGKeyboardKeyEventArgs e);

public: void OnKeyUp(CGKeyboardKeyEventArgs e);

public: void OnMouseWheelMoved(CGMouseWheelEventArgs e);
	//Not using the accept functionality for windows because it is not needed
public: void OnWindowFocusedChanged(FocusEventArgs e);

public: void OnWindowClosing(CGCancelEventArgs e);

public: void OnWindowClosed();

		void onFileDrop(FileDropEvent ev);
};
