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
#include "sw.h"

/*thread_local*/ static std::vector<sw::Entry> times;
typedef std::chrono::high_resolution_clock::time_point TimePoint;
/*thread_local*/ static int indent = 0;

static double getElapsedMilliseconds(TimePoint tp)
{
	return (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tp)).count();
}

/*void sw::start() { startTime = std::chrono::high_resolution_clock::now(); }

void sw::printElapsed(string desc) {
	cout << desc << " took " << getElapsedMilliseconds() << "ms" << endl;
}*/

void sw::timeit(string desc, std::function<void()> func) {
	auto startTime = std::chrono::high_resolution_clock::now();
	indent++;
	func();
	indent--;
	Entry entry;
	entry.index = times.size();
	entry.desc = desc;
	entry.elapsed = getElapsedMilliseconds(startTime);
	entry.indent = ::indent;
	times.push_back(entry);
}

void sw::beginFrame() {
	times.clear();
}

void sw::endFrame() {
	//cout << "=== TIMINGS ===" << endl;
	for (auto& time : times) {
		for (int i = 0; i < time.indent; i++) {
			cout << "\t";
		}
		cout << time.desc << " took " << time.elapsed << "ms" << endl;
	}
}
