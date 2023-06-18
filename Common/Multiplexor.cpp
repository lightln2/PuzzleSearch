#include "Multiplexor.h"
#include "StreamVInt.h"

SimpleMultiplexor::SimpleMultiplexor(Store& store, int segmentsCount) 
    : m_Store(store)
    , m_Lengths(segmentsCount, 0)
    , m_Buffers(segmentsCount * BUFSIZE)
{}

void SimpleMultiplexor::Add(int segment, uint32_t value) {
    size_t offset = segment * BUFSIZE;
    auto& len = m_Lengths[segment];
    m_Buffers[offset + len] = value;
    ++len;
    if (len == BUFSIZE) {
        Flush(segment);
    }
}

void SimpleMultiplexor::FlushAllSegments() {
    for (int i = 0; i < m_Lengths.size(); i++) {
        Flush(i);
    }
}

void SimpleMultiplexor::Flush(int segment) {
    auto& len = m_Lengths[segment];
    if (len == 0) return;
    size_t offset = segment * BUFSIZE;
    m_Store.WriteArray(segment, &m_Buffers[offset], len);
    len = 0;
}

Multiplexor::Multiplexor(StoreSet& storeSet, int segmentsCount)
{
    for (auto& store : storeSet.Stores) {
        m_Mults.emplace_back(SimpleMultiplexor(store, segmentsCount));
    }
}

void Multiplexor::Add(int op, int segment, uint32_t value) {
    m_Mults[op].Add(segment, value);
}

void Multiplexor::FlushAllSegments() {
    for (auto& mult : m_Mults) {
        mult.FlushAllSegments();
    }
}

CompressedMultiplexorPart::CompressedMultiplexorPart(Store& store, int segmentsCount,Buffer<uint8_t>& buffer)
    : m_Store(store)
    , m_Lengths(segmentsCount, 0)
    , m_Buffers(BUFSIZE * segmentsCount)
    , m_CompressedBuffer(buffer)
{}

void CompressedMultiplexorPart::Add(int segment, uint32_t value) {
    size_t offset = segment * BUFSIZE;
    auto& len = m_Lengths[segment];
    m_Buffers[offset + len] = value;
    ++len;
    if (len == BUFSIZE) {
        Flush(segment);
    }
}

void CompressedMultiplexorPart::Flush(int segment) {
    auto& len = m_Lengths[segment];
    if (len == 0) return;
    size_t offset = segment * BUFSIZE;
    int compressed = StreamVInt::Encode(len, &m_Buffers[offset], &m_CompressedBuffer[0], m_CompressedBuffer.Capacity());
    m_Store.WriteArray(segment, &m_CompressedBuffer[0], compressed);
    len = 0;
}

void CompressedMultiplexorPart::FlushAllSegments() {
    for (int i = 0; i < m_Lengths.size(); i++) {
        Flush(i);
    }
}

CompressedMultiplexor::CompressedMultiplexor(StoreSet& storeSet, int segmentsCount)
    : m_CompressedBuffer(StreamVInt::MAX_BUFFER_SIZE)
{
    for (auto& store : storeSet.Stores) {
        m_Mults.emplace_back(CompressedMultiplexorPart(store, segmentsCount, m_CompressedBuffer));
    }
}

void CompressedMultiplexor::Add(int op, int segment, uint32_t value) {
    m_Mults[op].Add(segment, value);
}

void CompressedMultiplexor::FlushAllSegments() {
    for (auto& mult : m_Mults) {
        mult.FlushAllSegments();
    }
}

SmartMultiplexorPart::SmartMultiplexorPart(
    Store& store,
    int segmentsCount,
    int mapSectorSizeBits,
    int smallSectorValsBits,
    Buffer<uint8_t>& largeBuffer,
    Buffer<uint8_t>& encodeBuffer)
    : m_Store(store)
    , m_Map(mapSectorSizeBits, std::max(segmentsCount, 4))
    , m_Files(segmentsCount, {m_Map})
    , m_SegmentsCount(segmentsCount)
    , m_SmallSectorValsBits(smallSectorValsBits)
    , m_SmallSectorSize(1ui64 << smallSectorValsBits)
    , m_SmallBuffer(m_SegmentsCount << smallSectorValsBits)
    , m_BufLengths(segmentsCount, 0)
    , m_LargeBuffer(largeBuffer)
    , m_EncodeBuffer(encodeBuffer)
    , m_MaxSectorsCount((largeBuffer.Capacity() + m_Map.GetSectorSize() - 1) / m_Map.GetSectorSize())
    , m_UsedList(segmentsCount)
{
    ensure(m_SmallSectorSize <= StreamVInt::MAX_INDEXES_COUNT);
}

void SmartMultiplexorPart::Add(int segment, uint32_t value) {
    auto& len = m_BufLengths[segment];
    size_t offset = size_t(segment) << m_SmallSectorValsBits;
    m_SmallBuffer[offset + len] = value;
    len++;
    if (len == m_SmallSectorSize) {
        FlushBuffer(segment);
    }
}

void SmartMultiplexorPart::FlushBuffer(int segment) {
    auto& len = m_BufLengths[segment];
    if (len == 0) return;
    size_t offset = size_t(segment) << m_SmallSectorValsBits;
    int compressed = StreamVInt::Encode(len, &m_SmallBuffer[offset], &m_EncodeBuffer[0], m_EncodeBuffer.Capacity());
    auto& file = m_Files[segment];
    if (!file.CanWriteWithoutExpand(compressed) && file.GetSectorsCount() >= m_MaxSectorsCount) {
        FlushFile(segment);
    }
    if (!file.CanWriteWithoutExpand(compressed) && !m_Map.HasFreeSectors()) {
        ensure(m_UsedList.head() != -1);
        //std::cerr 
        //    << "Flushing old file: seg=" << m_UsedList.head()
        //    << "; size=" << m_Files[m_UsedList.head()].TotalSize()
        //    << std::endl;
        FlushFile(m_UsedList.head());
        ensure(m_Map.HasFreeSectors());
    }

    if (m_Files[segment].IsEmpty()) {
        m_UsedList.add(segment);
    }

    m_Files[segment].Write(&m_EncodeBuffer[0], compressed);
    len = 0;
}

void SmartMultiplexorPart::FlushFile(int segment) {
    auto& file = m_Files[segment];
    if (file.IsEmpty()) return;
    size_t size = file.Read(&m_LargeBuffer[0], m_LargeBuffer.Capacity());
    file.Clear();
    m_Store.WriteArray(segment, &m_LargeBuffer[0], size);
    m_UsedList.remove(segment);
}

void SmartMultiplexorPart::FlushAllBuffers() {
    for (size_t i = 0; i < m_BufLengths.size(); i++) {
        FlushBuffer(i);
        FlushFile(i);
    }
}

void SmartMultiplexorPart::FlushAllSegments() {
    FlushAllBuffers();
}

SmartMultiplexor::SmartMultiplexor(
    StoreSet& storeSet,
    int maxOpBits,
    int segmentsCount,
    int mapSectorSizeBits,
    int smallSectorValsBits,
    size_t largeBufferSize)
    : m_StoreSet(storeSet)
    , m_Mults(maxOpBits)
    , m_SegmentsCount(segmentsCount)
    , m_MapSectotSizeBits(mapSectorSizeBits)
    , m_SmallSectorValsBits(smallSectorValsBits)
    , m_LargeBuffer(largeBufferSize)
    , m_EncodeBuffer(StreamVInt::MAX_BUFFER_SIZE)
{
}

void SmartMultiplexor::Add(int op, int segment, uint32_t value) {
    if (!m_Mults[op].get()) {
        m_Mults[op] = std::make_unique<SmartMultiplexorPart>(
            m_StoreSet.Stores[op],
            m_SegmentsCount,
            m_MapSectotSizeBits,
            m_SmallSectorValsBits,
            m_LargeBuffer,
            m_EncodeBuffer
        );
    }
    m_Mults[op]->Add(segment, value);
}

void SmartMultiplexor::FlushAllSegments() {
    for (auto& mult : m_Mults) {
        if (mult.get()) {
            mult->FlushAllSegments();
        }
    }
}

