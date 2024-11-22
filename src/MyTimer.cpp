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
#include "MyTimer.h"

vector<MyTimer*> TimerManager::timers;
vector<MyTimer*> TimerManager::newTimers;

MyTimer::MyTimer()
{
	TimerManager::registerTimer(this);
}

void MyTimer::setInterval(double sec)
{
	this->interval = sec;
}

void MyTimer::start()
{
	timerImpl.start();
	startedOnFrame = ci::app::getElapsedFrames();
}

bool MyTimer::isStopped() const
{
	return timerImpl.isStopped();
}

double MyTimer::getSeconds() const
{
	return timerImpl.getSeconds();
}

void MyTimer::singleShot(double intervalSec, function<void()> callback)
{
	auto timer = new MyTimer;
	timer->setInterval(intervalSec);
	//timer->timeout.connect(callback);
	timer->start();
}

void TimerManager::update()
{
	auto timersCopy = timers;
	vector<MyTimer*> leftTimers;
	for (MyTimer* timer : timersCopy) {
		if (timer->isStopped())
			continue;
		bool shouldFireTimeout = timer->getSeconds() > timer->interval && app::getElapsedFrames() != timer->startedOnFrame && timer->interval != 0;
		shouldFireTimeout |= (app::getElapsedFrames() == timer->startedOnFrame + 2) && timer->interval == 0;
		if (shouldFireTimeout) {
			//timer->timeout();
			delete timer;
		} else {
			leftTimers.push_back(timer);
		}
	}
	timers = leftTimers;
	timers.insert(timers.end(), newTimers.begin(), newTimers.end());
	newTimers.clear();
}

void TimerManager::registerTimer(MyTimer* timer)
{
	newTimers.push_back(timer);
}
