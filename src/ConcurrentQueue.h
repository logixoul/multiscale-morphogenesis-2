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

// https://codereview.stackexchange.com/a/177655
template<typename Item>
class ConcurrentQueue {
private:
	std::mutex mut;
	std::queue<Item> queue;
	std::condition_variable cv;

public:
	ConcurrentQueue() : queue() {}

	bool try_pop(Item& item, bool blocking) {
		//cout << "[try_pop] type=" << typeid(Item).name() << ", isEmpty" <<  << endl;
		std::unique_lock<std::mutex> lock(mut);
		if (blocking) {
			cv.wait(lock, [&]() { return !queue.empty(); });
		}
		if (queue.empty()) return false;
		item = queue.front();
		queue.pop();
		//cout << "[" << std::this_thread::get_id() << "][popped] size=" << queue.size() << endl;
		return true;
	}

	void push(Item item) { // todo: this was taking a rvalue reference
		{
		std::unique_lock<std::mutex> lock(mut);
		queue.push(item); // todo: this was using std::move
		}
		cv.notify_all();
		//cout << "[" << std::this_thread::get_id() << "][pushed] size=" << queue.size() << endl;
	}
};

