#pragma once

#include "GpuSolver.h"
#include "Util.h"

template<int width, int height>
class MTVerticalMoves {
    static constexpr int MAX_CROSS_SEGMENT = HostBuffer::SIZE;
    static constexpr int MAX_SAME_SEGMENT = HostBuffer::SIZE / (height - 1);
public:
    void SetSegment(int segment) {
        ensure(m_BufferPosition == 0);
        m_Segment = segment;
    }

    int GetSegment() const { return m_Segment; }

    uint32_t* GetBuffer() const { return m_Buffer.Buffer; }
    size_t GetBufferSize() const { return m_BufferPosition; }
    uint32_t* GetSegmentBuffer() const { return m_SegmentsBuffer.Buffer; }

    void Clear() { m_BufferPosition = 0; }

    bool AddSameSegment(int index) {
        m_Buffer.Buffer[m_BufferPosition++] = index;
        return m_BufferPosition == MAX_SAME_SEGMENT;
    }

    bool AddCrossSegment(int index) {
        m_Buffer.Buffer[m_BufferPosition++] = index;
        return m_BufferPosition == MAX_CROSS_SEGMENT;
    }

    size_t VertSameSegment() {
        if (m_BufferPosition == 0) return 0;
        m_GpuSolver.MTVertSameSegment(m_Segment, m_Buffer.Buffer, m_BufferPosition);
        return m_BufferPosition * (height - 1);
    }

    size_t VertCrossSegment() {
        if (m_BufferPosition == 0) return 0;
        m_GpuSolver.MTVertCrossSegment(m_Segment, m_Buffer.Buffer, m_SegmentsBuffer.Buffer, m_BufferPosition);
        return m_BufferPosition;
    }

private:
    int m_Segment;

    GpuSolver<width, height> m_GpuSolver;
    HostBuffer m_Buffer;
    HostBuffer m_SegmentsBuffer;
    size_t m_BufferPosition = 0;
};
