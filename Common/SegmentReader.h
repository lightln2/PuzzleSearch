#pragma once

#include "Buffer.h"
#include "Store.h"

#include <vector>

class SegmentReader {
    static constexpr size_t BUFSIZE = 2 * 1024 * 1024;

public:
    SegmentReader(Store& store);

    void SetSegment(int segment);

    void Delete(int segment);

    Buffer<uint32_t>& Read();

private:
    Store& m_Store;
    int m_Segment = -1;
    Buffer<uint32_t> m_Buffer;
};

class OpBitsReader {
    static constexpr size_t BUFSIZE = 2 * 1024 * 1024;

public:
    OpBitsReader(Store& store);

    void SetSegment(int segment);

    void Delete(int segment);

    Buffer<uint8_t>& Read();

private:
    Store& m_Store;
    int m_Segment = -1;
    Buffer<uint8_t> m_Buffer;
};

class CrossSegmentReader {
    static constexpr size_t BUFSIZE = 2 * 1024 * 1024;

public:
    CrossSegmentReader(StoreSet& storeSet);

    void SetSegment(int segment);

    Buffer<uint32_t>& Read(int op);

    void Delete(int segment);

private:
    StoreSet& m_StoreSet;
    int m_Segment = -1;
    Buffer<uint32_t> m_Buffer;
};

class FrontierReader {
    static constexpr size_t BUFSIZE = 2 * 1024 * 1024;
public:
    struct FrontierBuffer {
        Buffer<uint32_t>& Indexes;
        Buffer<uint8_t>& OpBits;

        bool IsEmpty() const { return Indexes.IsEmpty(); }
        size_t Size() const { return Indexes.Size(); }
    };

public:
    FrontierReader(Store& store);

    void SetSegment(int segment);

    void Delete(int segment);

    FrontierBuffer Read();

private:
    Store& m_Store;
    int m_Segment = -1;
    Buffer<uint32_t> m_IndexBuffer;
    Buffer<uint8_t> m_OpsBuffer;
};
