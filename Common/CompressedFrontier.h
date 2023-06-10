#pragma once

#include "Buffer.h"
#include "Store.h"
#include "StreamVInt.h"

class CompressedFrontierReader {
    static constexpr size_t BUFSIZE = 12 * 1024 * 1024;
public:
    struct FrontierBuffer {
        Buffer<uint32_t>& Indexes;
        Buffer<uint8_t>& OpBits;

        bool IsEmpty() const { return Indexes.IsEmpty(); }
        size_t Size() const { return Indexes.Size(); }
    };

public:
    CompressedFrontierReader(Store& store);

    void SetSegment(int segment);

    void Delete(int segment);

    FrontierBuffer Read();

private:
    Store& m_Store;
    int m_Segment = -1;
    Buffer<uint8_t> m_InputBuffer;
    size_t m_InputPos;
    Buffer<uint32_t> m_IndexBuffer;
    Buffer<uint8_t> m_OpsBuffer;
};

class CompressedFrontierWriter {
    static constexpr size_t BUFSIZE = 12 * 1024 * 1024;

public:
    CompressedFrontierWriter(Store& store);

    void SetSegment(int segment);

    void Add(uint32_t value, int opBits);

    void Flush();

private:
    void FlushData();
    void FlushBuffer();

private:
    Store& m_Store;
    int m_Segment = -1;
    Buffer<uint32_t> m_IndexBuffer;
    Buffer<uint8_t> m_OpsBuffer;
    Buffer<uint8_t> m_CompressedBuffer;
    Buffer<uint8_t> m_OutputBuffer;
};

class CompressedCrossSegmentReader {
    static constexpr size_t BUFSIZE = 2 * 1024 * 1024;

public:
    CompressedCrossSegmentReader(StoreSet& storeSet);

    void SetSegment(int segment);

    Buffer<uint32_t>& Read(int op);

    void Delete(int segment);

private:
    StoreSet& m_StoreSet;
    int m_Segment = -1;
    int m_LastOp = -1;
    Buffer<uint8_t> m_InputBuffer;
    size_t m_InputPos;
    Buffer<uint32_t> m_IndexBuffer;
};

