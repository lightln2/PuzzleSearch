#pragma once

#include "Util.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace file {
    using FHANDLE = void*;

    FHANDLE OpenFile(const std::string& fileName);
    void CloseFile(FHANDLE fd);
    void DeleteFile(const std::string& fileName);
    void CreateDirectory(const std::string& directory);
    void DeleteDirectory(const std::string& directory);

    void Seek(FHANDLE fd, uint64_t offset);
    size_t Read(FHANDLE fd, void* buffer, size_t size);
    void Write(FHANDLE fd, void* buffer, size_t size);
    size_t Read(FHANDLE fd, void* buffer, uint64_t offset, size_t size);
    void Write(FHANDLE fd, void* buffer, uint64_t offset, size_t size);

}

class RWFile {
public:
    RWFile(const std::string& fileName)
        : m_FileName(fileName)
        , m_Handle(file::OpenFile(fileName))
    { }

    RWFile() = default;
    RWFile(RWFile&) = delete;
    RWFile(const RWFile&) = delete;
    RWFile& operator=(RWFile&) = delete;
    RWFile& operator=(const RWFile&) = delete;
    RWFile(RWFile&&) = default;
    RWFile& operator=(RWFile&&) = default;

    ~RWFile() {
        if (!m_Handle) return;
        file::CloseFile(m_Handle);
        file::DeleteFile(m_FileName);
    }

    const std::string& FileName() const { return m_FileName; }

    void Rewind() { file::Seek(m_Handle, 0); }
    void Seek(uint64_t offset) { file::Seek(m_Handle, offset); }

    void Write(void* buffer, size_t size) { file::Write(m_Handle, buffer, size); }
    void Write(void* buffer, uint64_t offset, size_t size) { file::Write(m_Handle, buffer, offset, size); }

    size_t Read(void* buffer, size_t size) { return file::Read(m_Handle, buffer, size); }
    size_t Read(void* buffer, uint64_t offset, size_t size) { return file::Read(m_Handle, buffer, offset, size); }

    void Recreate() {
        if (m_Handle) {
            file::CloseFile(m_Handle);
            file::DeleteFile(m_FileName);
        }
        m_Handle = file::OpenFile(m_FileName);
    }

private:
    file::FHANDLE m_Handle = nullptr;
    std::string m_FileName;
};
