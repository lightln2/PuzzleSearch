#pragma once

#include "Array.h"

class SectorMap {
public:
    struct LinkedList {
        int head = -1;
        int tail = -1;
    };

public:
    SectorMap(int sectorSizeBits, int sectorsCount);

    void Free(LinkedList& list);

    bool HasFreeSectors() const { return m_FreeList.head != -1; }

    void GetFreeSector(LinkedList& list);

    size_t GetSectorSize() const { return m_SectorSize; }

    int GetSectorSizeBits() const { return m_SectorSizeBits; }

    size_t GetSectorsCount() const { return m_SectorsCount; }

    uint8_t* GetSector(size_t index) { return &m_Buffer[index << m_SectorSizeBits]; }

    const uint8_t* GetSector(size_t index) const { return &m_Buffer[index << m_SectorSizeBits]; }

    int& Next(int sector) { return m_NextSectorList[sector]; }

    const int& Next(int sector) const { return m_NextSectorList[sector]; }

private:
    int m_SectorSizeBits;
    int m_SectorsCount;
    size_t m_SectorSize;
    TArray<uint8_t> m_Buffer;
    std::vector<int> m_NextSectorList;
    LinkedList m_FreeList;
};

class SectorFile {
public:
    SectorFile(SectorMap& map)
        : m_Map(map)
        , m_SectorsCount(0)
        , m_LastPtr(m_Map.GetSectorSize())
    {}

    bool IsEmpty() const { return m_SectorsCount == 0; }

    size_t TotalSize() const { 
        return ((m_SectorsCount - 1) << m_Map.GetSectorSizeBits()) + m_LastPtr;
    }

    size_t GetSectorsCount() const { return m_SectorsCount; }

    bool CanWriteWithoutExpand(size_t size) const {
        ensure(size <= m_Map.GetSectorSize());
        return m_LastPtr + size <= m_Map.GetSectorSize();
    }

    void Write(uint8_t* buffer, size_t size) {
        size_t copy1 = std::min(m_Map.GetSectorSize() - m_LastPtr, size);
        CopyExistingTo(buffer, copy1);
        size_t copy2 = size - copy1;
        if (copy2 > 0) {
            AddNewSector();
            CopyExistingTo(buffer + copy1, copy2);
        }
    }

    size_t Read(uint8_t* buffer, size_t capacty) {
        ensure(TotalSize() <= capacty);
        uint8_t* dst = buffer;
        for (int ptr = m_List.head; ptr != m_List.tail; ptr = m_Map.Next(ptr)) {
            memcpy(dst, m_Map.GetSector(ptr), m_Map.GetSectorSize());
            dst += m_Map.GetSectorSize();
        }
        memcpy(dst, m_Map.GetSector(m_List.tail), m_LastPtr);
        return size_t(dst + m_LastPtr - buffer);
    }

    void Clear() {
        m_Map.Free(m_List);
        m_SectorsCount = 0;
        m_LastPtr = m_Map.GetSectorSize();
    }

private:
    void CopyExistingTo(uint8_t* buffer, size_t size) {
        assert(m_LastPtr + size <= m_Map.GetSectorSize());
        if (size == 0) return;
        uint8_t* lastSector = m_Map.GetSector(m_List.tail);
        memcpy(lastSector + m_LastPtr, buffer, size);
        m_LastPtr += size;
    }

    void AddNewSector() {
        assert(m_LastPtr = m_Map.GetSectorSize());
        m_Map.GetFreeSector(m_List);
        m_LastPtr = 0;
        m_SectorsCount++;
    }

private:
    SectorMap& m_Map;
    SectorMap::LinkedList m_List;
    size_t m_SectorsCount;
    size_t m_LastPtr;
};