#include "File.h"

#include <windows.h>

#include "Util.h"

namespace file {

FHANDLE OpenFile(const std::string& fileName) {
    HANDLE fd = CreateFileA(
        fileName.c_str(),
        GENERIC_WRITE | GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE,
        //FILE_ATTRIBUTE_NORMAL,
        NULL);
    ensure(fd != INVALID_HANDLE_VALUE);
    return fd;
}

void CloseFile(FHANDLE fd) {
    ensure(CloseHandle(fd));
}

#undef DeleteFile
void DeleteFile(const std::string& fileName) {
    DeleteFileA(fileName.c_str());
}

size_t Read(FHANDLE fd, void* buffer, size_t size) {
    DWORD actualSize = 0;
    ensure(size < 2ULL * 1024 * 1024 * 1024);
    ensure(ReadFile(fd, buffer, (DWORD)size, &actualSize, NULL));
    return actualSize;
}

void Write(FHANDLE fd, void* buffer, size_t size) {
    DWORD actualSize = 0;
    ensure(size < 2ULL * 1024 * 1024 * 1024);
    ensure(WriteFile(fd, buffer, (DWORD)size, &actualSize, NULL));
    ensure(actualSize == size);
}

void SeekBeginning(FHANDLE fd) {
    SetFilePointer(fd, 0, 0, FILE_BEGIN);
}


#undef CreateDirectory
void CreateDirectory(const std::string& directory) {
    CreateDirectoryA(directory.c_str(), NULL);
}

void DeleteDirectory(const std::string& directory) {
    RemoveDirectoryA(directory.c_str());
}

} // namespace file
