#pragma once
class System
{
public:
	static bool isMouseButtonHeld();
private:
	System();
};

class DialogBoxes
{
public:
	static vector<string> getOpenFilesPath(bool multipleSelection, bool* cancelled);
	static string getSaveFilePath(bool* cancelled, string defaultExt);
};
