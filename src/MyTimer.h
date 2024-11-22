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

class MyTimer
{
public:
	enum Unit { Seconds, Frames };
	MyTimer();

	void setInterval(double sec);
	void start();

	double interval;

	bool isStopped() const;

	double getSeconds() const;

	// TODO
	//signal<void()> timeout;

	static void singleShot(double intervalSec, function<void()> callback);
	
private:
	ci::Timer timerImpl;
	Unit unit;
	int startedOnFrame;
	friend class TimerManager;
};

class TimerManager {
public:
	static void update();

private:
	static void registerTimer(MyTimer* timer);
	static vector<MyTimer*> timers;
	static vector<MyTimer*> newTimers;
	friend class MyTimer;
};