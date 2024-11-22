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
#include "ConcurrentQueue.h"

class CrossThreadCallQueue
{
public:
	CrossThreadCallQueue();
	void pushCall(function<void()>&& func);
	void execAll();

private:
	//std::thread::id owningThread;
	ConcurrentQueue<function<void()>> toExecute;
};

extern thread_local CrossThreadCallQueue crossThreadCallQueue;

///////////////// UNRELATED CODE FOLLOWS

// https://stackoverflow.com/questions/14489935/implementing-futurethen-equivalent-for-asynchronous-execution-in-c11
template <typename T, typename Work>
auto then(future<T> f, Work w) -> future<decltype(w(f.get()))>
{
	return async([](future<T> f, Work w)
	{ return w(f.get()); }, move(f), move(w));
}

template <typename Work>
auto then(future<void> f, Work w) -> future<decltype(w())>
{
	return async([](future<void> f, Work w)
	{ f.wait(); return w(); }, move(f), move(w));
}