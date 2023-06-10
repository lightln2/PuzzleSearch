#pragma once

#include "Util.h"
#include "Store.h"

#include <vector>

class SimpleMultiplexor {
    static constexpr size_t BUFSIZE = 16 * 1024;

public:
    SimpleMultiplexor(Store& store, int segmentsCount);

    void Add(int segment, uint32_t value);

    void FlushAllSegments();

private:
    void Flush(int segment);

private:
    Store& m_Store;
    std::vector<uint32_t> m_Lengths;
    std::vector<uint32_t> m_Buffers;
};


class Multiplexor {
    static constexpr size_t BUFSIZE = 16 * 1024;

public:
    Multiplexor(StoreSet& storeSet, int segmentsCount);

    void Add(int op, int segment, uint32_t value);

    void FlushAllSegments();

private:
    std::vector<SimpleMultiplexor> m_Mults;
};

class CompressedMultiplexorPart {
    static constexpr size_t BUFSIZE = 16 * 1024;

public:
    CompressedMultiplexorPart(Store& store, int segmentsCount, Buffer<uint8_t>& buffer);

    void Add(int segment, uint32_t value);

    void FlushAllSegments();

private:
    void Flush(int segment);

private:
    Store& m_Store;
    std::vector<uint32_t> m_Lengths;
    std::vector<uint32_t> m_Buffers;
    Buffer<uint8_t>& m_CompressedBuffer;
};

class CompressedMultiplexor {
    static constexpr size_t BUFSIZE = 16 * 1024;

public:
    CompressedMultiplexor(StoreSet& storeSet, int segmentsCount);

    void Add(int op, int segment, uint32_t value);

    void FlushAllSegments();

private:
    Buffer<uint8_t> m_CompressedBuffer;
    std::vector<CompressedMultiplexorPart> m_Mults;
};
