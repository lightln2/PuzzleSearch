#pragma once

#include "SegmentedFile.h"
#include "StreamVInt.h"

#include <cassert>

class MTFrontierFileWriter {
    static constexpr size_t FILE_BUFFER_SIZE = 16 * 1024 * 1024;
public:
    MTFrontierFileWriter(SegmentedFile& file)
        : m_File(file)
        , m_Indexes(StreamVInt::MAX_INDEXES_COUNT)
        , m_Buffer(FILE_BUFFER_SIZE)
    {}

    void SetSegment(int segment) {
        m_Segment = segment;
        ensure(m_Indexes.IsEmpty());
        ensure(m_Buffer.IsEmpty());
    }

    int GetSegment() const { return m_Segment; }

    void FinishSegment() {
        if (!m_Indexes.IsEmpty()) {
            FlushData();
        }
        if (!m_Buffer.IsEmpty()) {
            FlushBuffer();
        }
    }

    void Add(uint32_t index) {
        m_Indexes.Add(index);
        if (m_Indexes.IsFull()) {
            FlushData();
            if (m_Buffer.Size() + StreamVInt::MAX_BUFFER_SIZE > m_Buffer.Capacity()) {
                FlushBuffer();
            }
        }
    }

private:
    void FlushData() {
        StreamVInt::Encode(m_Indexes, m_Buffer);
        m_Indexes.Clear();
    }

    void FlushBuffer() {
        m_File.Write(m_Segment, m_Buffer);
        m_Buffer.Clear();
    }

private:
    int m_Segment = -1;
    SegmentedFile& m_File;
    Buffer<uint8_t> m_Buffer;
    Buffer<uint32_t> m_Indexes;
};

class MTFrontierFileReader {
public:
    static constexpr size_t FILE_BUFFER_SIZE = 16 * 1024 * 1024;

    MTFrontierFileReader(SegmentedFile& file)
        : m_File(file)
        , m_Buffer(FILE_BUFFER_SIZE)
        , m_Indexes(StreamVInt::MAX_INDEXES_COUNT)
    {}

    void SetSegment(int segment) {
        m_Segment = segment;
        m_File.Rewind(segment);
        m_Buffer.Clear();
        m_Position = 0;
    }

    int GetSegment() const { return m_Segment; }

    Buffer<uint32_t>& Read() {
        if (m_Position == m_Buffer.Size()) {
            m_File.Read(m_Segment, m_Buffer);
            m_Position = 0;
        }
        m_Indexes.Clear();
        m_Position = StreamVInt::Decode(m_Position, m_Buffer, m_Indexes);
        return m_Indexes;
    }

private:
    int m_Segment = -1;
    SegmentedFile& m_File;
    Buffer<uint8_t> m_Buffer;
    int m_Position = 0;
    Buffer<uint32_t> m_Indexes;
};
