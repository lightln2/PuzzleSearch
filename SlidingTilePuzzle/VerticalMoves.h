#pragma once

#include "GpuSolver.h"
#include "Util.h"

template<int width, int height>
class VerticalMoves {
public:
    void SetSegment(int segment) {
        ensure(m_UpBufferPosition == 0);
        ensure(m_DownBufferPosition == 0);
        m_Segment = segment;
    }

    int GetSegment() const { return m_Segment; }

    uint32_t* GetUpBuffer() const { return m_UpBuffer.Buffer; }
    size_t GetUpBufferSize() const { return m_UpBufferPosition; }
    uint32_t* GetDownBuffer() const { return m_DownBuffer.Buffer; }
    size_t GetDownBufferSize() const { return m_DownBufferPosition; }
    uint32_t* GetSegmentBuffer() const { return m_SegmentsBuffer.Buffer; }

    void ClearUp() { m_UpBufferPosition = 0; }

    void ClearDown() { m_DownBufferPosition = 0; }

    bool AddUp(int index) {
        m_UpBuffer.Buffer[m_UpBufferPosition++] = index;
        return m_UpBufferPosition == m_UpBuffer.SIZE;
    }

    bool AddDown(int index) {
        m_DownBuffer.Buffer[m_DownBufferPosition++] = index;
        return m_DownBufferPosition == m_UpBuffer.SIZE;
    }

    void UpSameSegment() {
        if (m_UpBufferPosition == 0) return;
        m_GpuSolver.GpuUpSameSegment(m_Segment, m_UpBuffer.Buffer, m_UpBufferPosition);
    }

    void DownSameSegment() {
        if (m_DownBufferPosition == 0) return;
        m_GpuSolver.GpuDownSameSegment(m_Segment, m_DownBuffer.Buffer, m_DownBufferPosition);
    }

    void Up() {
        if (m_UpBufferPosition == 0) return;
        m_GpuSolver.GpuUp(m_Segment, m_UpBuffer.Buffer, m_SegmentsBuffer.Buffer, m_UpBufferPosition);
    }

    void Down() {
        if (m_DownBufferPosition == 0) return;
        m_GpuSolver.GpuDown(m_Segment, m_DownBuffer.Buffer, m_SegmentsBuffer.Buffer, m_DownBufferPosition);
    }

private:
    int m_Segment;

    GpuSolver<width, height> m_GpuSolver;
    HostBuffer m_UpBuffer;
    size_t m_UpBufferPosition = 0;
    HostBuffer m_DownBuffer;
    size_t m_DownBufferPosition = 0;
    HostBuffer m_SegmentsBuffer;

};