#pragma once

#include "Util.h"
#include "SectorMap.h"
#include "Store.h"

#include <vector>

class SimpleMultiplexor {
    static constexpr size_t BUFSIZE = 4 * 1024;

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

class SmartMultiplexorPart {

public:
    SmartMultiplexorPart(
        Store& store,
        int segmentsCount,
        int mapSectorSizeBits,
        int smallSectorValsBits,
        Buffer<uint8_t>& largeBuffer,
        Buffer<uint8_t>& encodeBuffer);

    void Add(int segment, uint32_t value);

    void FlushAllSegments();

private:
    void FlushBuffer(int segment);
    void FlushFile(int segment);
    void FlushAllBuffers();

private:
    Store& m_Store;
    SectorMap m_Map;

    size_t m_SegmentsCount;
    int m_SmallSectorValsBits;
    size_t m_SmallSectorSize;
    size_t m_MaxSectorsCount;

    std::vector<SectorFile> m_Files;
    Buffer<uint32_t> m_SmallBuffer;
    std::vector<int> m_BufLengths;
    Buffer<uint8_t>& m_LargeBuffer;
    Buffer<uint8_t>& m_EncodeBuffer;

    // list of files in order of creation
    DblLinkedListInt m_UsedList;
};

class SmartMultiplexor {
public:
    SmartMultiplexor(
        StoreSet& storeSet,
        int maxOpBits,
        int segmentsCount,
        int mapSectorSizeBits,
        int smallSectorValsBits,
        size_t largeBufferSize);

    void Add(int op, int segment, uint32_t value);

    void FlushAllSegments();

private:
    StoreSet& m_StoreSet;
    std::vector<std::unique_ptr<SmartMultiplexorPart>> m_Mults;

    size_t m_SegmentsCount;
    int m_MapSectotSizeBits;
    int m_SmallSectorValsBits;

    Buffer<uint8_t> m_LargeBuffer;
    Buffer<uint8_t> m_EncodeBuffer;
};
