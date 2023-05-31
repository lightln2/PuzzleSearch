#pragma once

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
        , m_Capacity(capacity)
        , m_Buffer((T*)malloc(capacity * sizeof(T)))
    {
        ensure(m_Buffer != nullptr);
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator =(const Buffer&) = delete;

    ~Buffer() { free(m_Buffer); }

    size_t Capacity() const { return m_Capacity; }

    void SetCapacity(size_t capacity) {
        free(m_Buffer);
        m_Buffer = (T*)malloc(capacity * sizeof(T));
        ensure(m_Buffer != nullptr);
    }

    size_t Size() const { return m_Size; }

    void SetSize(size_t size) { CHK(size - 1); m_Size = size; }

    T& operator[] (size_t index) { CHK(index); return m_Buffer[index]; }

    const T& operator[] (size_t index) const { CHK(index); return m_Buffer[index]; }

    void Add(T value) { CHK(m_Size); m_Buffer[m_Size++] = value; }

    void Clear() { m_Size = 0; }

    bool IsFull() const { CHK(m_Size - 1); return m_Size == m_Capacity; }

    bool IsEmpty() const { return m_Size == 0; }

    T* Buf() { return m_Buffer; }

    const T* Buf() const { return m_Buffer; }

    void SetAllZero() { memset(m_Buffer, 0, m_Size * sizeof(T)); }

    void CHK(size_t index) const {
#ifdef _DEBUG
        ensure(index < m_Capacity);
#endif
    }

private:
    size_t m_Size;
    size_t m_Capacity;
    T* m_Buffer;
};
