#pragma once

#include "SimplePuzzle.h"
#include "../SlidingTilePuzzle/File.h"

#include <cstdint>
#include <string>
#include <mutex>
#include <vector>

std::vector<uint64_t> DiskBasedTwoBitBFS(SimpleSlidingPuzzle& puzzle, std::string initialState);

class SimpleFileRW {
public:
    SimpleFileRW(const std::string& fileName);

    void Write(uint8_t* buffer, size_t count);
    size_t Read(uint8_t* buffer, size_t count);
    void Delete();
    void Rewind() { m_ReadHeader = 0; }

private:
    std::string m_FileName;
    std::unique_ptr<RWFile> m_File;
    std::mutex m_Mutex;
    uint64_t m_TotalLength = 0;
    int m_ReadHeader = 0;
    std::vector<uint64_t> m_Offsets;
    std::vector<uint64_t> m_Lengths;
};

class SimpleSegmentedFileRW {
public:
    SimpleSegmentedFileRW(const std::string& fileName) : m_FileName(fileName) {}

    void Write(int segment, uint8_t* buffer, size_t count) { GetFile(segment).Write(buffer, count); }
    size_t Read(int segment, uint8_t* buffer, size_t count) { GetFile(segment).Read(buffer, count); }
    void Delete(int segment) { GetFile(segment).Delete(); }
    void Rewind(int segment) { GetFile(segment).Rewind(); }

    template<typename T>
    void Write(int segment, std::vector<T>& vector) { 
        Write(segment, (uint8_t*)&vector[0], vector.size() * sizeof(T));
    }

    template<typename T>
    void Read(int segment, std::vector<T>& vector) {
        auto read = Read(segment, (uint8_t*)&vector[0], vector.capacity() * sizeof(T));
        vector.resize(read / sizeof(T));
    }

private:
    std::string GetFileName(int segment) { return m_FileName + std::to_string(segment); }
    SimpleFileRW& GetFile(int segment);
private:
    std::string m_FileName;
    std::vector<std::unique_ptr<SimpleFileRW>> m_Files;
};
