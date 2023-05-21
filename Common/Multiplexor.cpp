#include "Multiplexor.h"

Multiplexor::Multiplexor(Store& store, int segmentsCount) 
    : m_Store(store)
    , m_Lengths(segmentsCount, 0)
    , m_Buffers(segmentsCount * BUFSIZE)
{}

void Multiplexor::Add(int segment, uint32_t value) {
    size_t offset = segment * BUFSIZE;
    auto& len = m_Lengths[segment];
    m_Buffers[offset + len] = value;
    ++len;
    if (len == BUFSIZE) {
        Flush(segment);
    }
}

void Multiplexor::FlushAllSegments() {
    for (int i = 0; i < m_Lengths.size(); i++) {
        Flush(i);
    }
}

void Multiplexor::Flush(int segment) {
    auto& len = m_Lengths[segment];
    if (len == 0) return;
    size_t offset = segment * BUFSIZE;
    m_Store.WriteArray(segment, &m_Buffers[offset], len);
    len = 0;
}
