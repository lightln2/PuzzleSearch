#pragma once

#include "Array.h"

class SectorMap {
    struct LinkedList {
        int head = -1;
        int tail = -1;
    };
public:
    SectorMap(int sectorSizeBits, int sectorsCount);

    void Free(LinkedList& list);

    bool HasFreeSectors() const { return m_FreeList.head != -1; }

    void GetFreeSector(LinkedList& list);

private:
    int& Next(int sector) { return m_NextSectorList[sector]; }

    const int& Next(int sector) const { return m_NextSectorList[sector]; }

private:
    int m_SectorSizeBits;
    size_t m_SectorSize;
    int m_SectorsCount;
    TArray<uint32_t> m_Buffer;
    std::vector<int> m_NextSectorList;
    LinkedList m_FreeList;
};