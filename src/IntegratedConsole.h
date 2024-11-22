#pragma once

class IntegratedConsole
{
public:
	IntegratedConsole();
	void update();

private:
	std::stringstream redirectStream;
};

