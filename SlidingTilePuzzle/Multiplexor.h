#pragma once

#include "FrontierFile.h"
#include "Util.h"

#include <vector>
#include <optional>

class Multiplexor {
    static constexpr size_t BUFFER_SIZE = 4 * 1024 * 1024;
    static constexpr size_t SMALL_BUFFER_SIZE = 8 * 1024;
public:
    Multiplexor(int maxSegments, SegmentedFile& file)
        : m_Segments(maxSegments)
        , m_Writer(file, maxSegments)
    {
        m_UsedBuffers.reserve(64);
    }

    void Add(uint32_t segment, uint32_t index) {
        auto& writer = m_Segments[segment];
        if (!writer.has_value()) {
            writer.emplace(m_Writer);
            writer->SetSegment(segment);
            m_UsedBuffers.push_back(segment);
        }
        writer->Add(index);
    }

    void Close() {
        for (int segment : m_UsedBuffers) {
            FlushBuffer(segment);
            m_Segments[segment] = std::nullopt;
        }
        m_UsedBuffers.clear();
    }

    void CloseAll() {
        m_Writer.FlushAll();
    }

private:
    void FlushBuffer(int segment) {
        auto& writer = *m_Segments[segment];
        writer.FinishSegment();
    }

private:
    SmallSegmentWriter m_Writer;
    std::vector<std::optional<ExpandedFrontierWriter>> m_Segments;
    std::vector<int> m_UsedBuffers;
   
};