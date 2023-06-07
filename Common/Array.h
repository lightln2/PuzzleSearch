#pragma once

#include "Util.h"

#include <memory>

/* Simple array with resizing */
template <typename T>
class TArray {
public:

    TArray()
        : m_Size(0)
        , m_Buffer(nullptr)
    {}

    TArray(size_t size)
        : m_Size(size)
        , m_Buffer((T*)malloc(size * sizeof(T)))
    {
        ensure(m_Buffer != nullptr);
        memset(m_Buffer, 0, size * sizeof(T));
    }

    TArray(const TArray&) = delete;
    TArray& operator =(const TArray&) = delete;

    TArray(TArray&& other) noexcept {
        std::swap(m_Buffer, other.m_Buffer);
        std::swap(m_Size, other.m_Size);
    }

    TArray& operator =(TArray&& other) noexcept {
        std::swap(m_Buffer, other.m_Buffer);
        std::swap(m_Size, other.m_Size);
        return *this;
    }

    ~TArray() { if (m_Buffer) free(m_Buffer); }

    size_t Size() const { return m_Size; }

    void SetSize(size_t size) {
        m_Size = size;
        if (m_Buffer) free(m_Buffer);
        m_Buffer = (T*)malloc(size * sizeof(T));
        ensure(m_Buffer != nullptr);
        memset(m_Buffer, 0, size * sizeof(T));
    }

    T& operator[] (size_t index) noexcept { CHK(index); return m_Buffer[index]; }

    const T& operator[] (size_t index) const noexcept { CHK(index); return m_Buffer[index]; }

    T* Buf() { return m_Buffer; }

    const T* Buf() const { return m_Buffer; }

    void Clear() { memset(m_Buffer, 0, m_Size * sizeof(T)); }

private:
    void CHK(size_t index) const {
#ifdef _DEBUG
        ensure(index < m_Size);
#endif
    }

private:
    size_t m_Size;
    T* m_Buffer;
};
