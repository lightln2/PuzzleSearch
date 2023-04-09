#pragma once

#include "SegmentedFile.h"
#include "StreamVInt.h"

#include <cassert>

struct FrontierBuffer {
    size_t Count;
    uint32_t* Indexes;
    uint8_t* Bounds;
};

class FrontierFileWriter {
    static constexpr size_t FILE_BUFFER_SIZE = 16 * 1024 * 1024;
public:
    FrontierFileWriter(SegmentedFile& file)
        : m_File(file)
        , m_Buffer(FILE_BUFFER_SIZE)
        , m_Indexes(StreamVInt::MAX_INDEXES_COUNT)
        , m_Bounds(StreamVInt::MAX_INDEXES_COUNT)
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

    void Add(uint32_t index, uint8_t bounds) {
        m_Indexes.Add(index);
        m_Bounds.Add(bounds);
        if (m_Indexes.IsFull()) {
            FlushData();
            if (m_Buffer.Size() + StreamVInt::MAX_BUFFER_SIZE > m_Buffer.Capacity()) {
                FlushBuffer();
            }
        }
    }

private:
    void FlushData() {
        StreamVInt::Encode(m_Indexes, m_Bounds, m_Buffer);
        m_Indexes.Clear();
        m_Bounds.Clear();
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
    Buffer<uint8_t> m_Bounds;
};

class FrontierFileReader {
public:
    static constexpr size_t FILE_BUFFER_SIZE = 16 * 1024 * 1024;

    FrontierFileReader(SegmentedFile& file)
        : m_File(file)
        , m_Buffer(FILE_BUFFER_SIZE)
        , m_Indexes(StreamVInt::MAX_INDEXES_COUNT)
        , m_Bounds(StreamVInt::MAX_INDEXES_COUNT)
    {}

    void SetSegment(int segment) {
        m_Segment = segment;
        m_File.Rewind(segment);
        m_Buffer.Clear();
        m_Position = 0;
    }

    int GetSegment() const { return m_Segment; }

    FrontierBuffer Read();

private:
    int m_Segment = -1;
    SegmentedFile& m_File;
    Buffer<uint8_t> m_Buffer;
    int m_Position = 0;
    Buffer<uint32_t> m_Indexes;
    Buffer<uint8_t> m_Bounds;
};

class SmallSegmentWriter {
    static constexpr size_t LARGE_BUFFER_SIZE = 8 * 1024 * 1024;
    static constexpr size_t SMALL_BUFFER_SIZE = 8 * 1024;
public:
    SmallSegmentWriter(SegmentedFile& file, int maxSegments);
    ~SmallSegmentWriter();

    void Consume(int segment, uint8_t* buf, int size);
    void FlushAll();
    
    Buffer<uint32_t>& GetIndexArray();
    Buffer<uint8_t>& GetBufferArray();

    void ResetPools() {
        m_IndexesPoolPos = 0;
        m_BuffersPoolPos = 0;
    }

private:
    SegmentedFile& m_File;
    uint8_t* m_Buffer;
    std::vector<int> m_Lengths;
   
    std::vector<std::unique_ptr<Buffer<uint32_t>>> m_IndexesPool;
    std::vector<std::unique_ptr<Buffer<uint8_t>>> m_BuffersPool;
    int m_IndexesPoolPos = 0;
    int m_BuffersPoolPos = 0;
};

class ExpandedFrontierWriter {
public:
    ExpandedFrontierWriter(SmallSegmentWriter& writer)
        : m_Writer(writer)
        , m_Indexes(writer.GetIndexArray())
        , m_Buffer(writer.GetBufferArray()) {}

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
        m_Writer.Consume(m_Segment, m_Buffer.Buf(), m_Buffer.Size());
        m_Buffer.Clear();
    }

private:
    int m_Segment = -1;
    SmallSegmentWriter& m_Writer;
    Buffer<uint32_t>& m_Indexes;
    Buffer<uint8_t>& m_Buffer;
};

class ExpandedFrontierReader {
public:
    static constexpr size_t BUFFER_SIZE = 8 * 1024 * 1024;

    ExpandedFrontierReader(SegmentedFile& file)
        : m_File(file)
        , m_Indexes(StreamVInt::MAX_INDEXES_COUNT)
        , m_Buffer(BUFFER_SIZE) {}

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

    void DeleteSegment() {
        m_File.Delete(m_Segment);
    }
private:
    int m_Segment = -1;
    int m_Position = 0;
    SegmentedFile& m_File;
    Buffer<uint32_t> m_Indexes;
    Buffer<uint8_t> m_Buffer;
};
