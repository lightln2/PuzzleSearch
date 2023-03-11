#pragma once

#include "FrontierFile.h"

#include <atomic>

class Collector {
public:
    Collector(size_t count, SegmentedFile& file);

    void SetSegment(uint32_t segment);
    void Add(uint32_t index, uint8_t bounds);

    size_t SaveSegment();

    void PrintStats();

private:
    SegmentedFile& m_File;
    FrontierFileWriter m_FrontierWriter;
    Buffer<uint8_t> m_Bounds;

private:
    //static std::atomic<uint64_t> m_NanosSaveSegment;
    static uint64_t m_NanosSaveSegment;
};