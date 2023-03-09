#pragma once

#include "SegmentedFile.h"

#include <array>

class FrontierFileWriter {
    static const size_t BUFFER_SIZE = 4 * 1024 * 1024;
public:
    FrontierFileWriter() : m_Buffer(BUFFER_SIZE) {}

    void SetSegment(SegmentedFile* file, int segment) {
        m_File = file;
        m_Segment = segment;
    }

    void FinishSegment() { FlushBuffer(); }

    void Add(uint32_t index, uint8_t bounds) {
        m_Buffer.Add(index);
        m_Buffer.Add(bounds);
        if (m_Buffer.IsFull()) FlushBuffer();
    }

private:
    void FlushBuffer() {
        if (m_Buffer.Size() == 0) return;
        m_File->Write(m_Segment, m_Buffer);
        m_Buffer.Clear();
    }

private:
    int m_Segment = -1;
    SegmentedFile* m_File = nullptr;
    Buffer<uint32_t> m_Buffer;
};

struct FrontierBuffer {
    size_t Count;
    uint32_t* Indexes;
    uint8_t* Bounds;
};

class FrontierFileReader {
public:
    static const size_t BUFFER_SIZE = 4 * 1024 * 1024;

    FrontierFileReader()
        : m_Buffer(BUFFER_SIZE)
        , m_Indexes(BUFFER_SIZE)
        , m_Bounds(BUFFER_SIZE)
    {}

    void SetSegment(SegmentedFile* file, int segment) {
        m_File = file;
        m_Segment = segment;
    }

    FrontierBuffer Read() {
        m_File->Read(m_Segment, m_Buffer);
        m_Indexes.Clear();
        m_Bounds.Clear();
        for (int i = 0; i < m_Buffer.Size(); i += 2) {
            m_Indexes.Add(m_Buffer[i]);
            m_Bounds.Add(m_Buffer[i + 1]);
        }
        return { m_Indexes.Size(), m_Indexes.Buf(), m_Bounds.Buf() };
    }

private:
    int m_Segment = -1;
    SegmentedFile* m_File = nullptr;
    Buffer<uint32_t> m_Buffer;
    Buffer<uint32_t> m_Indexes;
    Buffer<uint8_t> m_Bounds;
};


class ExpandedFrontierWriter {
    static const size_t BUFFER_SIZE = 4 * 1024 * 1024;
public:
    ExpandedFrontierWriter() : m_Buffer(BUFFER_SIZE) {}

    void SetSegment(SegmentedFile* file, int segment) {
        m_File = file;
        m_Segment = segment;
    }

    void FinishSegment() { FlushBuffer(); }

    void Add(uint32_t index) {
        m_Buffer.Add(index);
        if (m_Buffer.IsFull()) FlushBuffer();
    }

private:
    void FlushBuffer() {
        if (m_Buffer.Size() == 0) return;
        m_File->Write(m_Segment, m_Buffer);
        m_Buffer.Clear();
    }

private:
    int m_Segment = -1;
    SegmentedFile* m_File = nullptr;
    Buffer<uint32_t> m_Buffer;
};

class ExpandedFrontierReader {
public:
    static const size_t BUFFER_SIZE = 4 * 1024 * 1024;

    ExpandedFrontierReader() : m_Buffer(BUFFER_SIZE) {}

    void SetSegment(SegmentedFile* file, int segment) {
        m_File = file;
        m_Segment = segment;
    }

    Buffer<uint32_t>& Read() {
        m_File->Read(m_Segment, m_Buffer);
        return m_Buffer;
    }

private:
    int m_Segment = -1;
    SegmentedFile* m_File = nullptr;
    Buffer<uint32_t> m_Buffer;
};
