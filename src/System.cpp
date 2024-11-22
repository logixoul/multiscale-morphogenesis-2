#include "precompiled.h"
#include "System.h"
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <shobjidl.h> 
#include <cinder/msw/CinderMsw.h>

bool System::isMouseButtonHeld() {
	if ((GetKeyState(VK_LBUTTON) & 0x80) != 0)
	{
		return true;
	}
	if ((GetKeyState(VK_RBUTTON) & 0x80) != 0)
	{
		return true;
	}
	if ((GetKeyState(VK_MBUTTON) & 0x80) != 0)
	{
		return true;
	}
	return false;
}

vector<string> DialogBoxes::getOpenFilesPath(bool multipleSelection, bool* cancelled)
{
	*cancelled = false;
	vector<string> res;

	HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED |
		COINIT_DISABLE_OLE1DDE);
	if (SUCCEEDED(hr))
	{
		IFileOpenDialog *pFileOpen;

		// Create the FileOpenDialog object.
		hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
			IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));
		if (multipleSelection) {
			DWORD dwOptions;
			hr = pFileOpen->GetOptions(&dwOptions);
			hr = pFileOpen->SetOptions(dwOptions | FOS_ALLOWMULTISELECT);
		}
		if (SUCCEEDED(hr))
		{
			// Show the Open dialog box.
			hr = pFileOpen->Show(NULL);
			if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
				*cancelled = true;
			}
			// Get the file name from the dialog box.
			if (SUCCEEDED(hr))
			{
				//IShellItem *pItem;
				//hr = pFileOpen->GetResult(&pItem);
				if (SUCCEEDED(hr))
				{
					PWSTR pszFilePath;
					//hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

					if (SUCCEEDED(hr))
					{
						//res = pszFilePath;
						IShellItemArray *psiaResult;
						hr = pFileOpen->GetResults(&psiaResult);
						PWSTR pszFilePath = NULL;
						DWORD dwNumItems = 0;

						hr = psiaResult->GetCount(&dwNumItems);
						for (DWORD i = 0; i < dwNumItems; i++) {
							IShellItem *psi = NULL;

							hr = psiaResult->GetItemAt(i, &psi);
							hr = psi->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);
							std::wstring str16 = pszFilePath;
							res.push_back(ci::msw::toUtf8String(str16));
							CoTaskMemFree(pszFilePath);
							psi->Release();
						}
						psiaResult->Release();
					}
					//pItem->Release();
				}
			}
			pFileOpen->Release();
		}
		CoUninitialize();
	}
	return res;
}

string DialogBoxes::getSaveFilePath(bool* cancelled, string defaultExt) {
	*cancelled = false;
	string res;

	auto hwnd = (HWND)getWindow()->getNative();
	IFileSaveDialog *pfd;
	HRESULT hr = CoCreateInstance(CLSID_FileSaveDialog,
		NULL,
		CLSCTX_INPROC_SERVER,
		IID_PPV_ARGS(&pfd));

	COMDLG_FILTERSPEC fileTypes[] =
	{
		{L"PNG Files", L"*.png"},
		{L"JPEG Files", L"*.jpg"},
		{L"TIFF Files", L"*.tiff"},
	};

	if (SUCCEEDED(hr))
	{
		hr = pfd->SetFileTypes(3, fileTypes);
		pfd->SetDefaultExtension(msw::toWideString(defaultExt).c_str());
		if (SUCCEEDED(hr))
		{
			// Show the dialog
			hr = pfd->Show(hwnd);

			if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
				*cancelled = true;
			}
			if (SUCCEEDED(hr))
			{
				// Obtain the result of the user's interaction with the dialog.
				IShellItem *psiResult;
				hr = pfd->GetResult(&psiResult);
				if (SUCCEEDED(hr))
				{
					LPWSTR fileName;

					hr = psiResult->GetDisplayName(SIGDN_FILESYSPATH, &fileName);

					if (SUCCEEDED(hr))
					{
						wstring wstr = fileName;
						res = msw::toUtf8String(wstr);
						CoTaskMemFree(fileName);
					}
					psiResult->Release();
				}
			}
		}
		pfd->Release();
	}
	return res;
}
