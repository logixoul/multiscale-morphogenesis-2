#pragma once
#include "precompiled.h"

class QDebug {
public:
	~QDebug() {
		std::cout << "\n";
	}
};

template<class T>
QDebug& operator<<(QDebug& prev, T const& value) {
	std::cout << value;
	return prev;
}

QDebug qDebug();