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
