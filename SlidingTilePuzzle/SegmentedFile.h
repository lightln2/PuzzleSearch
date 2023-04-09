#pragma once

#include "File.h"
#include "Util.h"

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

class SegmentedFilePart {

    struct Chunk {
        uint64_t offset;
        uint32_t length;
        int next;
    };

public:
    SegmentedFilePart(int maxSegments, const std::string& filePath);

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

class SegmentedFile {
public:
    SegmentedFile(int maxSegments, const std::string& file)
        : SegmentedFile(maxSegments, { file })
    { }

    SegmentedFile(int maxSegments, std::initializer_list<std::string> files) 
        : SegmentedFile(maxSegments, std::vector<std::string>(files))
    {}

    SegmentedFile(int maxSegments, const std::vector<std::string>& files, bool sequentialFiles = false) 
        : m_SequentialFiles(sequentialFiles)
    {
        for (const auto& file : files) {
            m_Files.emplace_back(maxSegments, file);
        }
        m_SegmentToFile.reserve(maxSegments);
        if (sequentialFiles) {
            m_Lock = std::make_unique<std::mutex>();
            m_UndeletedSegments.resize(m_Files.size());
            for (int i = 0; i < maxSegments; i++) {
                int index = m_Files.size() * i / maxSegments;
                m_SegmentToFile.push_back(index);
                m_UndeletedSegments[index]++;
            }

        }
        else {
            for (int i = 0; i < maxSegments; i++) {
                m_SegmentToFile.push_back(i % m_Files.size());
            }
        }
    }

    bool HasData(int segment) const { return GetFile(segment).HasData(segment); }

    int MaxSegments() const { return GetFile(0).MaxSegments(); }

    uint64_t Length(int segment) const { return GetFile(segment).Length(segment); }

    uint64_t TotalLength() const { 
        uint64_t length = 0;
        for (const auto& file : m_Files) {
            length += file.TotalLength();
        }
        return length;
    }

    void Rewind(int segment) { GetFile(segment).Rewind(segment); }

    void RewindAll() {
        for (auto& file : m_Files) {
            file.RewindAll();
        }
    }

    void Write(int segment, void* buffer, size_t size) {
        GetFile(segment).Write(segment, buffer, size);
    }

    size_t Read(int segment, void* buffer, size_t size) {
        return GetFile(segment).Read(segment, buffer, size);
    }

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

    void Delete(int segment) {
        if (m_SequentialFiles) {
            int index = m_SegmentToFile[segment];
            m_Lock->lock();
            m_UndeletedSegments[index]--;
            if (m_UndeletedSegments[index] == 0) {
                //std::cerr << "Delete " << index << std::endl;
                m_Files[index].DeleteAll();
            }
            m_Lock->unlock();
        }
    }
    void DeleteAll() {
        for (auto& file : m_Files) {
            file.DeleteAll();
        }
        if (m_SequentialFiles) {
            for (int i = 0; i < m_UndeletedSegments.size(); i++) {
                m_UndeletedSegments[i] = 0;
            }
            for (int i = 0; i < m_SegmentToFile.size(); i++) {
                int index = m_SegmentToFile[i];
                m_UndeletedSegments[index]++;
            }
        }
    }

    static void PrintStats() {
        SegmentedFilePart::PrintStats();
    }

private:
    SegmentedFilePart& GetFile(int segment) {
        return m_Files[m_SegmentToFile[segment]];
    }

    const SegmentedFilePart& GetFile(int segment) const {
        return m_Files[m_SegmentToFile[segment]];
    }

private:
    bool m_SequentialFiles;
    std::vector<int> m_UndeletedSegments;
    std::vector<int> m_SegmentToFile;
    std::vector<SegmentedFilePart> m_Files;
    std::unique_ptr<std::mutex> m_Lock;
};