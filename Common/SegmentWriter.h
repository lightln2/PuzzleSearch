#pragma once

#include "Buffer.h"
#include "Store.h"

class SegmentWriter {
    static constexpr size_t BUFSIZE = 2 * 1024 * 1024;

public:
    SegmentWriter(Store& store);

    void SetSegment(int segment);

    void Add(uint32_t value);

    void Flush();

private:
    Store& m_Store;
    int m_Segment = -1;
    Buffer<uint32_t> m_Buffer;
};

class OpBitsWriter {
    static constexpr size_t BUFSIZE = 2 * 1024 * 1024;

public:
    OpBitsWriter(Store& store);

    void SetSegment(int segment);

    void Add(uint8_t value);

    void Flush();

private:
    Store& m_Store;
    int m_Segment = -1;
    Buffer<uint8_t> m_Buffer;
};

class FrontierWriter {
    static constexpr size_t BUFSIZE = 2 * 1024 * 1024;

public:
    FrontierWriter(Store& store);

    void SetSegment(int segment);

    void Add(uint32_t value, int opBits);

    void Flush();

private:
    Store& m_Store;
    int m_Segment = -1;
    Buffer<uint32_t> m_IndexBuffer;
    Buffer<uint8_t> m_OpsBuffer;
};
