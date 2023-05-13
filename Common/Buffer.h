#pragma once

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
