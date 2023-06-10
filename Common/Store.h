#pragma once

#include "Buffer.h"
#include "Util.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class StoreImpl {
public:
    virtual ~StoreImpl() noexcept {};

    virtual int MaxSegments() const = 0;
    virtual uint64_t Length(int segment) const = 0;
    virtual void Rewind(int segment) = 0;
    virtual void Write(int segment, void* buffer, size_t size) = 0;
    virtual size_t Read(int segment, void* buffer, size_t size) = 0;
    virtual void Delete(int segment) = 0;

    virtual bool HasData(int segment) const { return Length(segment) == 0; }

    virtual uint64_t TotalLength() const {
        uint64_t total = 0;
        for (int i = 0; i < MaxSegments(); i++) total += Length(i);
        return total;
    }

    virtual void RewindAll() {
        for (int i = 0; i < MaxSegments(); i++) Rewind(i);
    }

    virtual void DeleteAll() {
        for (int i = 0; i < MaxSegments(); i++) Delete(i);
    }
};

using StoreImplRef = std::unique_ptr<StoreImpl>;

class Store {
public:
    static Store CreateSingleFileStore(int maxSegments, const std::string& filePath);
    static Store CreateSingleFileStore(int maxSegments, std::vector<std::string> filePaths);
    static Store CreateMultiFileStore(int maxSegments, const std::string& directory);
    static Store CreateMultiFileStore(int maxSegments, std::vector<std::string> directories);
    static Store CreateSequentialStore(int maxSegments, std::vector<std::string> filePaths);

    static Store CreateMultiFileStore(int maxSegments,
                                      std::vector<std::string> directories,
                                      const std::string& subdir);
private:
    Store(StoreImplRef impl);

public:
    int MaxSegments() const;
    bool HasData(int segment) const;
    uint64_t Length(int segment) const;
    uint64_t TotalLength() const;

    void Rewind(int segment);
    void RewindAll();

    void Write(int segment, void* buffer, size_t size);
    size_t Read(int segment, void* buffer, size_t size);

    void Delete(int segment);
    void DeleteAll();

    static void PrintStats();

    template<typename T>
    void WriteArray(int segment, const T* buffer, size_t size) {
        Write(segment, (void*)buffer, size * sizeof(T));
    }

    template<typename T>
    void Write(int segment, const std::vector<T>& buffer) {
        WriteArray<T>(segment, &buffer[0], buffer.size());
    }

    template<typename T>
    void Write(int segment, const Buffer<T>& buffer) {
        WriteArray<T>(segment, buffer.Buf(), buffer.Size());
    }

    template<typename T>
    size_t ReadArray(int segment, T* buffer, size_t capacity) {
        size_t read = Read(segment, (void*)buffer, capacity * sizeof(T));
        ensure(read % sizeof(T) == 0);
        return read / sizeof(T);
    }

    template<typename T>
    void Read(int segment, std::vector<T>& buffer) {
        buffer.resize(buffer.capacity());
        size_t read = ReadArray<T>(segment, &buffer[0], buffer.capacity());
        buffer.resize(read);
    }

    template<typename T>
    void Read(int segment, Buffer<T>& buffer) {
        size_t read = ReadArray<T>(segment, buffer.Buf(), buffer.Capacity());
        buffer.SetSize(read);
    }

private:
    static std::vector<std::string> CreatePaths(
        const std::vector<std::string>& directories,
        const std::string& path);

private:
    StoreImplRef m_Impl;

private:
    static std::atomic<uint64_t> m_StatReadsCount;
    static std::atomic<uint64_t> m_StatReadNanos;
    static std::atomic<uint64_t> m_StatReadBytes;
    static std::atomic<uint64_t> m_StatWritesCount;
    static std::atomic<uint64_t> m_StatWriteNanos;
    static std::atomic<uint64_t> m_StatWriteBytes;
};

struct StoreSet {
    std::vector<Store> Stores;

    int MaxSegments() const { return Stores[0].MaxSegments(); }

    uint64_t TotalLength() const {
        uint64_t total = 0;
        for (const auto& store : Stores) {
            total += store.TotalLength();
        }
        return total;
    }

    void Rewind(int segment) {
        for (auto& store : Stores) {
            store.Rewind(segment);
        }
    }

    void RewindAll() {
        for (auto& store : Stores) {
            store.RewindAll();
        }
    }

    void Delete(int segment) {
        for (auto& store : Stores) {
            store.Delete(segment);
        }
    }

    void DeleteAll() {
        for (auto& store : Stores) {
            store.DeleteAll();
        }
    }

    void Swap(StoreSet& other) {
        for (int i = 0; i < Stores.size(); i++) {
            std::swap(Stores[i], other.Stores[i]);
        }
    }
};

