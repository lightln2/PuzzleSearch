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