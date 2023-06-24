#include "File.h"

#include <windows.h>

namespace file {

    FHANDLE OpenFile(const std::string& fileName) {
        HANDLE fd = CreateFileA(
            fileName.c_str(),
            GENERIC_WRITE | GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            NULL);
        if (fd == INVALID_HANDLE_VALUE) {
            std::cerr << "Cannot open file: " << fileName << std::endl;
            ensure(fd != INVALID_HANDLE_VALUE);
        }
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

    size_t Read(FHANDLE fd, void* buffer, uint64_t offset, size_t size) {
        DWORD actualSize = 0;
        ensure(size < 2ULL * 1024 * 1024 * 1024);
        OVERLAPPED over{ 0 };
        over.Offset = offset & 0xFFFFFFFF;
        over.OffsetHigh = offset >> 32;
        ensure(ReadFile(fd, buffer, (DWORD)size, &actualSize, &over));
        return actualSize;
    }

    void Write(FHANDLE fd, void* buffer, uint64_t offset, size_t size) {
        DWORD actualSize = 0;
        ensure(size < 2ULL * 1024 * 1024 * 1024);
        OVERLAPPED over{ 0 };
        over.Offset = offset & 0xFFFFFFFF;
        over.OffsetHigh = offset >> 32;
        ensure(WriteFile(fd, buffer, (DWORD)size, &actualSize, &over));
        ensure(actualSize == size);
    }


    void Seek(FHANDLE fd, uint64_t offset) {
        LONG offset_high = offset >> 32;
        ensure(SetFilePointer(fd, (LONG)offset, &offset_high, FILE_BEGIN) != INVALID_SET_FILE_POINTER);
    }


#undef CreateDirectory
    void CreateDirectory(const std::string& directory) {
        CreateDirectoryA(directory.c_str(), NULL);
    }

    void DeleteDirectory(const std::string& directory) {
        RemoveDirectoryA(directory.c_str());
    }

    bool FileExists(const std::string& fileName) {
        auto attrs = GetFileAttributesA(fileName.c_str());
        return attrs != INVALID_FILE_ATTRIBUTES;
    }

} // namespace file
