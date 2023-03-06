#pragma once

#include "File.h"

#include <memory>
#include <optional>
#include <vector>

class SegmentedFile {
public:
    SegmentedFile(int maxSegments, const std::string& directory);

    uint64_t Length(int segment) { return m_Sizes[segment]; }
    bool HasData(int segment) { return Length(segment) > 0; }

    uint64_t TotalLength() {
        uint64_t totalLength = 0;
        for (auto size : m_Sizes) totalLength += size;
        return totalLength;
    }

    void Write(int segment, void* buffer, size_t size);
    size_t Read(int segment, void* buffer, size_t size);
    void Delete(int segment);
    void DeleteAll();

private:
    std::string SegmentFileName(int segment);

private:
    std::string m_Directory;
    std::vector<std::optional<RWFile>> m_Files;
    std::vector<uint64_t> m_Sizes;
};
