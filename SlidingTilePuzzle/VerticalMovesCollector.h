#pragma once

#include "GpuSolver.h"
#include "Multiplexor.h"
#include "Puzzle.h"
#include "SegmentedFile.h"
#include "VerticalMoves.h"

template<int width, int height>
class VerticalMovesCollector {
public:
    VerticalMovesCollector(
        SegmentedFile& expandedUp,
        SegmentedFile& expandedDown,
        VerticalMoves<width, height>& verticalMoves)
        : m_VerticalMoves(verticalMoves)
        , m_MultUp(Puzzle<width, height>::MaxSegments(), expandedUp)
        , m_MultDown(Puzzle<width, height>::MaxSegments(), expandedDown)
    {}

    void SetSegment(uint32_t segment) {
        Close();
        m_VerticalMoves.SetSegment(segment);
    }

    void Add(size_t count, uint32_t* indexes, uint8_t* bounds) {
        for (size_t i = 0; i < count; i++) {
            Add(indexes[i], bounds[i]);
        }
    }

    void Add(int index, uint8_t bound) {
        int blank = index & 15;
        if (!(bound & Puzzle<width, height>::B_UP) && Puzzle<width, height>::UpChangesSegment(blank)) {
            AddUp(index);
        }
        if (!(bound & Puzzle<width, height>::B_DOWN) && Puzzle<width, height>::DownChangesSegment(blank)) {
            AddDown(index);
        }
    }

    void AddUp(uint32_t index) {
        if (m_VerticalMoves.AddUp(index)) {
            FlushUp();
        }
    }

    void AddDown(uint32_t index) {
        if (m_VerticalMoves.AddDown(index)) {
            FlushDown();
        }
    }

    void Close() {
        FlushUp();
        FlushDown();
        m_MultUp.Close();
        m_MultDown.Close();
    }

    void CloseAll() {
        m_MultUp.CloseAll();
        m_MultDown.CloseAll();
    }

private:
    void FlushUp() {
        m_VerticalMoves.Up();
        auto* buf = m_VerticalMoves.GetUpBuffer();
        auto* seg = m_VerticalMoves.GetSegmentBuffer();
        auto size = m_VerticalMoves.GetUpBufferSize();
        for (int i = 0; i < size; i++) {
            m_MultUp.Add(seg[i], buf[i]);
        }
        m_VerticalMoves.ClearUp();
    }

    void FlushDown() {
        m_VerticalMoves.Down();
        auto* buf = m_VerticalMoves.GetDownBuffer();
        auto* seg = m_VerticalMoves.GetSegmentBuffer();
        auto size = m_VerticalMoves.GetDownBufferSize();
        for (int i = 0; i < size; i++) {
            m_MultDown.Add(seg[i], buf[i]);
        }
        m_VerticalMoves.ClearDown();
    }

private:
    SimpleMultiplexor m_MultUp;
    SimpleMultiplexor m_MultDown;
    VerticalMoves<width, height>& m_VerticalMoves;
};