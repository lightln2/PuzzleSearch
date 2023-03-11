#pragma once

#include "GpuSolver.h"
#include "Multiplexor.h"
#include "Puzzle.h"
#include "SegmentedFile.h"

template<int width, int height>
class VerticalMovesCollector {
public:
    VerticalMovesCollector(SegmentedFile& expandedUp, SegmentedFile& expandedDown)
        : m_ExpandedUp(expandedUp)
        , m_ExpandedDown(expandedDown)
        , m_MultUp(Puzzle<width, height>::MaxSegments(), expandedUp)
        , m_MultDown(Puzzle<width, height>::MaxSegments(), expandedDown)
    {}

    void SetSegment(uint32_t segment) {
        m_MultUp.Close();
        m_MultDown.Close();
        m_Segment = segment;
    }

    void AddUp(uint32_t index) {
        m_BufferUp.Buffer[m_PositionUp++] = index;
        if (m_PositionUp == m_BufferUp.SIZE) FlushUp();
    }

    void AddDown(uint32_t index) {
        m_BufferDown.Buffer[m_PositionDown++] = index;
        if (m_PositionDown == m_BufferUp.SIZE) FlushDown();
    }

    void Close() {
        FlushUp();
        FlushDown();
        m_MultUp.Close();
        m_MultDown.Close();
    }

private:
    void FlushUp() {
        if (m_PositionUp == 0) return;
        for (int i = 0; i < m_PositionUp; i++) {
            auto [newsegment, newindex] = Puzzle<width, height>::MoveUp(m_Segment, m_BufferUp.Buffer[i]);
            m_BufferUp.Buffer[i] = newindex;
            m_BufferSegments.Buffer[i] = newsegment;
        }
        //m_GpuSolver.GpuUp(m_Segment, m_BufferUp.Buffer, m_BufferSegments.Buffer, m_PositionUp);
        for (int i = 0; i < m_PositionUp; i++) {
            m_MultUp.Add(m_BufferSegments.Buffer[i], m_BufferUp.Buffer[i]);
        }
        m_PositionUp = 0;
    }

    void FlushDown() {
        if (m_PositionDown == 0) return;
        for (int i = 0; i < m_PositionDown; i++) {
            auto [newsegment, newindex] = Puzzle<width, height>::MoveDown(m_Segment, m_BufferDown.Buffer[i]);
            m_BufferDown.Buffer[i] = newindex;
            m_BufferSegments.Buffer[i] = newsegment;
        }
        //m_GpuSolver.GpuDown(m_Segment, m_BufferDown.Buffer, m_BufferSegments.Buffer, m_PositionDown);
        for (int i = 0; i < m_PositionDown; i++) {
            m_MultDown.Add(m_BufferSegments.Buffer[i], m_BufferDown.Buffer[i]);
        }
        m_PositionDown = 0;
    }

private:
    uint32_t m_Segment = -1;
    SegmentedFile& m_ExpandedUp;
    SegmentedFile& m_ExpandedDown;
    Multiplexor m_MultUp;
    Multiplexor m_MultDown;
    GpuSolver<width, height> m_GpuSolver;
    HostBuffer m_BufferUp;
    HostBuffer m_BufferDown;
    HostBuffer m_BufferSegments;
    int m_PositionUp = 0;
    int m_PositionDown = 0;
};