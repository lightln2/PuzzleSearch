#pragma once

#include "GpuSolver.h"
#include "Multiplexor.h"
#include "Puzzle.h"
#include "SegmentedFile.h"
#include "MTVerticalMoves.h"

template<int width, int height>
class MTVerticalMovesCollector {
public:
    MTVerticalMovesCollector(
        SegmentedFile& expanded,
        MTVerticalMoves<width, height>& verticalMoves)
        : m_VerticalMoves(verticalMoves)
        , m_Mult(Puzzle<width, height>::MaxSegments(), expanded)
    {}

    void SetSegment(uint32_t segment) {
        Close();
        m_VerticalMoves.SetSegment(segment);
    }

    void Add(size_t count, uint32_t* indexes) {
        for (size_t i = 0; i < count; i++) {
            Add(indexes[i]);
        }
    }

    void Add(int index) {
        if (m_VerticalMoves.AddCrossSegment(index)) {
            Flush();
        }
    }

    void Close() {
        Flush();
        m_Mult.Close();
    }

    void CloseAll() {
        m_Mult.CloseAll();
    }

private:
    void Flush() {
        auto size = m_VerticalMoves.VertCrossSegment();
        auto* buf = m_VerticalMoves.GetBuffer();
        auto* seg = m_VerticalMoves.GetSegmentBuffer();
        for (int i = 0; i < size; i++) {
            if (buf[i] != uint32_t(-1)) {
                m_Mult.Add(seg[i], buf[i]);
            }
        }
        m_VerticalMoves.Clear();
    }

private:
    SimpleMultiplexor m_Mult;
    MTVerticalMoves<width, height>& m_VerticalMoves;
};