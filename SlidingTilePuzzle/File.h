#pragma once

#include <cstdint>
#include <iostream>
#include <string>

namespace file {
    using FHANDLE = void*;

    FHANDLE OpenFile(const std::string& fileName);
    void CloseFile(FHANDLE fd);
    void DeleteFile(const std::string& fileName);
    uint32_t Read(FHANDLE fd, void* buffer, uint32_t size);
    void Write(FHANDLE fd, void* buffer, uint32_t size);
    void SeekBeginning(FHANDLE fd);
    void CreateDirectory(const std::string& directory);
}

class RWFile {
public:
    RWFile(const std::string& fileName) 
        : m_FileName(fileName)
        , m_Handle(file::OpenFile(fileName))
    { }

    RWFile(const RWFile&) = delete;
    RWFile& operator=(const RWFile&) = delete;

    ~RWFile() {
        file::CloseFile(m_Handle);
        file::DeleteFile(m_FileName);
    }

    void Write(void* buffer, uint32_t size) {
        file::Write(m_Handle, buffer, size);
    }

    uint32_t Read(void* buffer, uint32_t size) {
        if (m_IsWriting) {
            m_IsWriting = false;
            file::SeekBeginning(m_Handle);
        }
        return file::Read(m_Handle, buffer, size);
    }

private:
    file::FHANDLE m_Handle;
    std::string m_FileName;
    bool m_IsWriting = true;
};
