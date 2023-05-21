#pragma once

#include "BoolArray.h"
#include "Store.h"

#include <vector>

class SegmentReader {
    static constexpr size_t BUFSIZE = 2 * 1024 * 1024;

public:
    SegmentReader(Store& store);

    void SetSegment(int segment);

    std::vector<uint32_t>& Read();


private:
    Store& m_Store;
    int m_Segment = -1;
    std::vector<uint32_t> m_Buffer;
};