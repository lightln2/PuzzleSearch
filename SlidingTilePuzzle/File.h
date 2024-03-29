#pragma once

#include <cstdint>
#include <iostream>
#include <string>

#include "Util.h"

namespace file {
    using FHANDLE = void*;

    FHANDLE OpenFile(const std::string& fileName);
    void CloseFile(FHANDLE fd);
    void DeleteFile(const std::string& fileName);
    size_t Read(FHANDLE fd, void* buffer, size_t size);
    void Write(FHANDLE fd, void* buffer, size_t size);
    void Seek(FHANDLE fd, uint64_t offset);
    void CreateDirectory(const std::string& directory);
    void DeleteDirectory(const std::string& directory);

    size_t Read(FHANDLE fd, void* buffer, uint64_t offset, size_t size);
    void Write(FHANDLE fd, void* buffer, uint64_t offset, size_t size);

}

template <typename T>
class Buffer {
public:
    Buffer(size_t capacity)
        : m_Size(0)
        , m_Capacity(capacity)
        , m_Buffer((T*)malloc(capacity * sizeof(T)))
    {
        ensure(m_Buffer != nullptr);
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator =(const Buffer&) = delete;

    ~Buffer() {
        free(m_Buffer);
    }

    void Add(T value) { m_Buffer[m_Size++] = value; }
    void Clear() { m_Size = 0; }
    T& operator[] (size_t index) { return m_Buffer[index]; }
    const T& operator[] (size_t index) const { return m_Buffer[index]; }
    size_t Size() const { return m_Size; }
    void SetSize(size_t size) { m_Size = size; }
    size_t Capacity() const { return m_Capacity; }
    bool IsFull() const { return m_Size == m_Capacity; }
    bool IsEmpty() const { return m_Size == 0; }
    T* Buf() const { return m_Buffer; }

private:
    size_t m_Size;
    size_t m_Capacity;
    T* m_Buffer;
};


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

    void Rewind() {
        file::Seek(m_Handle, 0);
    }

    void Seek(uint64_t offset) {
        file::Seek(m_Handle, offset);
    }

    void Write(void* buffer, size_t size) {
        file::Write(m_Handle, buffer, size);
    }

    size_t Read(void* buffer, size_t size) {
        return file::Read(m_Handle, buffer, size);
    }

    void Write(void* buffer, uint64_t offset, size_t size) {
        file::Write(m_Handle, buffer, offset, size);
    }

    size_t Read(void* buffer, uint64_t offset, size_t size) {
        return file::Read(m_Handle, buffer, offset, size);
    }

    template<typename T>
    void Write(const Buffer<T>& buffer) {
        Write((void*)buffer.Buf(), buffer.Size() * sizeof(T));
    }

    template<typename T>
    void Read(Buffer<T>& buffer) {
        size_t read = Read(buffer.Buf(), buffer.Capacity() * sizeof(T));
        ensure(read % sizeof(T) == 0);
        buffer.SetSize(read / sizeof(T));
    }

private:
    file::FHANDLE m_Handle;
    std::string m_FileName;
};
