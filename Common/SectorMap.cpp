#include "SectorMap.h"

SectorMap::SectorMap(int sectorSizeBits, int sectorsCount)
    : m_SectorSizeBits(sectorSizeBits)
    , m_SectorsCount(sectorsCount)
    , m_SectorSize(1ui64 << sectorSizeBits)
    , m_Buffer(sectorsCount * m_SectorSize)
    , m_NextSectorList(sectorsCount, 0)
    , m_FreeList({0, sectorsCount - 1})
{
    for (int i = 0; i < m_NextSectorList.size(); i++) {
        m_NextSectorList[i] = i + 1;
    }
    m_NextSectorList.back() = -1;
}

void SectorMap::Free(SectorMap::LinkedList& list) {
    if (list.head == -1) return;
    assert(list.tail != -1);
    if (m_FreeList.head == -1) {
        m_FreeList.head = list.head;
    }
    else {
        Next(m_FreeList.tail) = list.head;
    }
    m_FreeList.tail = list.tail;
    list.head = list.tail = -1;
}

void SectorMap::GetFreeSector(SectorMap::LinkedList& list) {
    ensure(HasFreeSectors());
    if (list.head == -1) {
        list.head = m_FreeList.head;
    }
    else {
        Next(list.tail) = m_FreeList.head;
    }
    list.tail = m_FreeList.head;
    m_FreeList.head = Next(m_FreeList.head);
    Next(list.tail) = -1;
}
