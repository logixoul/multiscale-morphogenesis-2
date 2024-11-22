#include "precompiled.h"
#include "IntegratedConsole.h"
#include <deque>
#include <sstream>
#include "CinderImGui.h"

IntegratedConsole::IntegratedConsole() {
	std::cout.rdbuf(redirectStream.rdbuf());
}

void IntegratedConsole::update()
{
	string curContents = redirectStream.str();
    istringstream curContentsStream(curContents);
    string line;
    deque<string> lines;
    while (std::getline(curContentsStream, line))
    {
        lines.push_back(line);
    }
    int numLines = lines.size();
    for (int i = 0; i < numLines - 10; i++) {
        lines.pop_front();
    }
    for (auto& line : lines) {
        ImGui::Text(line.c_str());
    }
}
