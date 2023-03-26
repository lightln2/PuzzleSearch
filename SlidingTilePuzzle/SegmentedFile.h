#pragma once

#include "File.h"
#include "Util.h"

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

class SegmentedFile {

    struct Chunk {
        uint64_t offset;
        uint32_t length;
        int next;
    };

public:
    SegmentedFile(int maxSegments, const std::string& filePath);

    bool HasData(int segment) const { return m_Heads[segment] > 0; }

    int MaxSegments() const { return (int)m_Heads.size(); }

    uint64_t Length(int segment) const;

    uint64_t TotalLength() const { return m_TotalLength; }

    void Rewind(int segment);

    void RewindAll();

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

    void DeleteAll();

    static void PrintStats();

private:
    std::string m_FilePath;
    std::unique_ptr<RWFile> m_File;
    uint64_t m_TotalLength;
    std::vector<Chunk> m_Chunks;
    std::vector<int> m_Heads;
    std::vector<int> m_Tails;
    std::vector<int> m_ReadPointers;
    std::unique_ptr<std::mutex> m_Mutex;

private:
    static std::atomic<uint64_t> m_StatReadsCount;
    static std::atomic<uint64_t> m_StatReadNanos;
    static std::atomic<uint64_t> m_StatReadBytes;
    static std::atomic<uint64_t> m_StatWritesCount;
    static std::atomic<uint64_t> m_StatWriteNanos;
    static std::atomic<uint64_t> m_StatWriteBytes;
};
