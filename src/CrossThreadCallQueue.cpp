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
#include "CrossThreadCallQueue.h"

// disabling all thread_local stuff because of RenderDoc. https://github.com/baldurk/renderdoc/issues/1743
//thread_local CrossThreadCallQueue crossThreadCallQueue;

CrossThreadCallQueue::CrossThreadCallQueue()
{
}

void CrossThreadCallQueue::pushCall(function<void()>&& func)
{
	toExecute.push(func);
}

void CrossThreadCallQueue::execAll()
{
	for (;;) {
		function<void()> func;
		if (toExecute.try_pop(func, false)) {
			func();
		}
		else {
			break;
		}
	}
}
