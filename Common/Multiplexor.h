#pragma once

#include "Util.h"
#include "Store.h"

#include <vector>

class SimpleMultiplexor {
    static constexpr size_t BUFSIZE = 16 * 1024;

public:
    SimpleMultiplexor(Store& store, int segmentsCount);

    void Add(int segment, uint32_t value);

    void FlushAllSegments();

private:
    void Flush(int segment);

private:
    Store& m_Store;
    std::vector<uint32_t> m_Lengths;
    std::vector<uint32_t> m_Buffers;
};