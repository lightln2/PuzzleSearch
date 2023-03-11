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
    {
        m_UsedBuffers.reserve(64);
    }

    void Add(uint32_t segment, uint32_t index) {
        auto& buffer = m_Segments[segment];
        if (!buffer.has_value()) {
            buffer.emplace(BUFFER_SIZE);
            m_UsedBuffers.push_back(segment);
        }
        buffer->Add(index);
        if (buffer->IsFull()) FlushBuffer(segment);
    }

    void Close() {
        for (int segment : m_UsedBuffers) {
            FlushBuffer(segment);
            m_Segments[segment] = std::nullopt;
        }
        m_UsedBuffers.clear();
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
    std::vector<int> m_UsedBuffers;
    SegmentedFile& m_File;
    ExpandedFrontierWriter m_ExpandedWriter;
};