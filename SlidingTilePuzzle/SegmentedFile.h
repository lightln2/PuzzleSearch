#pragma once

#include "File.h"
#include "Util.h"

#include <memory>
#include <optional>
#include <vector>

class SegmentedFile {
public:
    SegmentedFile(int maxSegments, const std::string& directory);

    uint64_t Length(int segment) const { return m_Sizes[segment]; }

    bool HasData(int segment) const { return Length(segment) > 0; }

    uint64_t TotalLength() const {
        uint64_t totalLength = 0;
        for (auto size : m_Sizes) totalLength += size;
        return totalLength;
    }

    void RewindAll();

    void Rewind(int segment);

    void Write(int segment, void* buffer, size_t size);

    size_t Read(int segment, void* buffer, size_t size);

    template<typename T> 
    void Write(int segment, Buffer<T>& buffer) {
        Write(segment, buffer.Buf(), buffer.Size() * sizeof(T));
    }

    template<typename T>
    void Read(int segment, Buffer<T>& buffer) {
        size_t read = Read(segment, buffer.Buf(), buffer.Capacity() * sizeof(T));
        ensure(read % sizeof(T) == 0);
        buffer.SetSize(read / sizeof(T));
    }

    void Delete(int segment);

    void DeleteAll();

private:
    std::string SegmentFileName(int segment);

private:
    std::string m_Directory;
    std::vector<std::optional<RWFile>> m_Files;
    std::vector<uint64_t> m_Sizes;
};
