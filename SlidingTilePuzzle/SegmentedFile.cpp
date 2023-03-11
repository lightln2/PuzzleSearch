#include "SegmentedFile.h"
#include "Util.h"

#include <cassert>
#include <iomanip>
#include <sstream>

SegmentedFile::SegmentedFile(int maxSegments, const std::string& directory)
    : m_Directory(directory)
    , m_Files(maxSegments)
    , m_Sizes(maxSegments)
{
    file::CreateDirectory(directory);
}

void SegmentedFile::Write(int segment, void* buffer, size_t size) {
    assert(segment >= 0 && segment < m_Files.size());
    auto& file = m_Files[segment];
    if (!file.has_value()) {
        auto fileName = SegmentFileName(segment);
        file.emplace(fileName);
    }
    file->Write(buffer, size);
    m_Sizes[segment] += size;
}

size_t SegmentedFile::Read(int segment, void* buffer, size_t size) {
    assert(segment >= 0 && segment < m_Files.size());
    auto& file = m_Files[segment];
    if (!file.has_value()) return 0;
    return file->Read(buffer, size);
}

void SegmentedFile::Delete(int segment) {
    m_Files[segment] = std::nullopt;
    m_Sizes[segment] = 0;
}

void SegmentedFile::DeleteAll() {
    for (int i = 0; i < m_Files.size(); i++) Delete(i);
}

std::string SegmentedFile::SegmentFileName(int segment) {
    std::ostringstream stream;
    stream << m_Directory << "/" << std::hex << std::setfill('0') << std::setw(3) << segment;
    return stream.str();
}

