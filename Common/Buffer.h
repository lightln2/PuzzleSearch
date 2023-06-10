#pragma once

#include "Array.h"
#include "Util.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

/* Simple implementation like std::vector but with constant capacity and resizing without initialization */
template <typename T>
class Buffer {
public:
    Buffer(size_t capacity)
        : m_Size(0)
        , m_Array(capacity)
    {}

    size_t Capacity() const { return m_Array.Size(); }

    void SetCapacity(size_t capacity) {
        m_Array.SetSize(capacity);
    }

    size_t Size() const { return m_Size; }

    void SetSize(size_t size) { ensure(m_Size <= Capacity()); m_Size = size; }

    T& operator[] (size_t index) { return m_Array[index]; }

    const T& operator[] (size_t index) const { return m_Array[index]; }

    void Add(T value) { m_Array[m_Size++] = value; }

    void Clear() { m_Size = 0; }

    bool IsFull() const { return m_Size == Capacity(); }

    bool IsEmpty() const { return m_Size == 0; }

    TArray<T>& Array() { return m_Array; }

    const TArray<T>& Array() const { return m_Array; }

    T* Buf() { return m_Array.Buf(); }

    const T* Buf() const { return m_Array.Buf(); }

    void SetAllZero() { m_Array.Clear(); }

    bool CanAppend(const Buffer<T>& buf) { return Size() + buf.Size() <= Capacity(); }

    void Append(const Buffer<T>& buf) {
        ensure(CanAppend(buf));
        memcpy(Buf() + Size(), buf.Buf(), buf.Size() * sizeof(T));
        m_Size += buf.Size();
    }

private:
    size_t m_Size;
    TArray<T> m_Array;
};
