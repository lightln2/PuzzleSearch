#pragma once

#include "FrontierFile.h"

class Collector {
public:
    Collector(size_t count, FrontierFileWriter& frontierWriter);

    void SetSegment(uint32_t segment);
    void Add(uint32_t index, uint8_t bounds);

    size_t SaveSegment();

private:
    FrontierFileWriter& m_FrontierWriter;
    Buffer<uint8_t> m_Bounds;
    uint32_t m_Segment;
};