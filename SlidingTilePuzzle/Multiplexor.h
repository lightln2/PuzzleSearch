#pragma once

#include "FrontierFile.h"
#include "Util.h"

#include <vector>
#include <optional>

class Multiplexor {
    static const size_t BUFFER_SIZE = 4 * 1024 * 1024;
public:
    Multiplexor(int maxSegments, SegmentedFile& file) 
        : m_Segments(maxSegments)
        , m_File(file)
        , m_ExpandedWriter(file)
    {}

    void Add(uint32_t segment, uint32_t index) {
        auto& buffer = m_Segments[segment];
        if (!buffer.has_value()) buffer.emplace(BUFFER_SIZE);
        buffer->Add(index);
        if (buffer->IsFull()) FlushBuffer(segment);
    }

    void Close() {
        for (int i = 0; i < m_Segments.size(); i++) {
            if (m_Segments[i].has_value()) FlushBuffer(i);
        }
    }

private:
    void FlushBuffer(int segment) {
        auto& buffer = *m_Segments[segment];
        m_ExpandedWriter.SetSegment(segment);
        for (int i = 0; i < buffer.Size(); i++) {
            m_ExpandedWriter.Add(buffer[i]);
        }
        m_ExpandedWriter.FinishSegment();
        buffer.Clear();
    }

private:
    std::vector<std::optional<Buffer<uint32_t>>> m_Segments;
    SegmentedFile& m_File;
    ExpandedFrontierWriter m_ExpandedWriter;
};