#include "File.h"
#include "Store.h"
#include "Util.h"

#include <mutex>
#include <optional>

std::atomic<uint64_t> Store::m_StatReadsCount(0);
std::atomic<uint64_t> Store::m_StatReadNanos(0);
std::atomic<uint64_t> Store::m_StatReadBytes(0);
std::atomic<uint64_t> Store::m_StatWritesCount(0);
std::atomic<uint64_t> Store::m_StatWriteNanos(0);
std::atomic<uint64_t> Store::m_StatWriteBytes(0);

Store::Store(StoreImplRef impl)
    : m_Impl(std::move(impl))
{}

int Store::MaxSegments() const {
    return m_Impl->MaxSegments();
}

bool Store::HasData(int segment) const {
    return m_Impl->HasData(segment);
}

uint64_t Store::Length(int segment) const {
    return m_Impl->Length(segment);
}

uint64_t Store::TotalLength() const {
    return m_Impl->TotalLength();
}

void Store::Rewind(int segment) {
    m_Impl->Rewind(segment);
}

void Store::RewindAll() {
    m_Impl->RewindAll();
}

void Store::Write(int segment, void* buffer, size_t size) {
    ensure(segment >= 0 && segment < MaxSegments());
    if (size == 0) return;
    m_StatWritesCount++;
    m_StatWriteBytes += size;
    Timer timer;
    m_Impl->Write(segment, buffer, size);
    m_StatWriteNanos += timer.Elapsed();
}

size_t Store::Read(int segment, void* buffer, size_t size) {
    ensure(segment >= 0 && segment < MaxSegments());
    Timer timer;
    size_t read = m_Impl->Read(segment, buffer, size);
    if (read > 0) {
        m_StatReadsCount++;
        m_StatReadBytes += read;
    }
    m_StatReadNanos += timer.Elapsed();
    return read;
}

void Store::Delete(int segment) {
    m_Impl->Delete(segment);
}

void Store::DeleteAll() {
    m_Impl->DeleteAll();
}

void Store::PrintStats() {
    std::cerr
        << "Store: "
        << "reads: " << WithDecSep(m_StatReadsCount)
        << "; " << WithSize(m_StatReadBytes) << " in " << WithTime(m_StatReadNanos)
        << "; writes: " << WithDecSep(m_StatWritesCount)
        << "; " << WithSize(m_StatWriteBytes) << " in " << WithTime(m_StatWriteNanos)
        << std::endl;
}

namespace {

    struct Chunk {
        uint64_t offset;
        uint32_t length;
        int next;
    };

} // namespace


class SingleFileStoreImpl : public StoreImpl {
public:
    SingleFileStoreImpl(int maxSegments, const std::string& filePath)
        : m_File(std::make_unique<RWFile>(filePath, false))
        , m_TotalLength(0)
        , m_NonemptySegments(0)
        , m_Chunks(0)
        , m_Heads(maxSegments, -1)
        , m_Tails(maxSegments, -1)
        , m_ReadPointers(maxSegments, -1)
        , m_Mutex(std::make_unique<std::mutex>())
    { }

    virtual ~SingleFileStoreImpl() noexcept {};

    virtual int MaxSegments() const { return (int)m_Heads.size(); }

    virtual bool HasData(int segment) const { return m_Heads[segment] >= 0; }

    virtual uint64_t Length(int segment) const {
        uint64_t totalLength = 0;
        int pos = m_Heads[segment];
        while (pos >= 0) {
            totalLength += m_Chunks[pos].length;
            pos = m_Chunks[pos].next;
        }
        return totalLength;
    }

    virtual void Rewind(int segment) {
        m_ReadPointers[segment] = m_Heads[segment];
    }

    virtual void Write(int segment, void* buffer, size_t size) {
        m_Mutex->lock();
        if (m_NonemptySegments == 0) m_File->Recreate();

        auto offset = m_TotalLength;
        m_TotalLength += size;
        m_File->Write(buffer, offset, size);
        int pos = (int)m_Chunks.size();
        m_Chunks.push_back(Chunk{ (uint64_t)offset, (uint32_t)size, -1 });

        if (m_Heads[segment] == -1) {
            m_Heads[segment] = m_Tails[segment] = m_ReadPointers[segment] = pos;
            m_NonemptySegments++;
        }
        else {
            m_Chunks[m_Tails[segment]].next = pos;
            m_Tails[segment] = pos;
        }
        m_Mutex->unlock();
    }

    virtual size_t Read(int segment, void* buffer, size_t size) {
        if (m_ReadPointers[segment] == -1) return 0;
        auto& chunk = m_Chunks[m_ReadPointers[segment]];
        ensure(chunk.length <= size);

        auto read = m_File->Read(buffer, chunk.offset, chunk.length);
        m_ReadPointers[segment] = chunk.next;
        return read;
    }

    virtual void Delete(int segment) {
        m_Mutex->lock();
        if (HasData(segment)) {
            m_Heads[segment] = m_Tails[segment] = m_ReadPointers[segment] = -1;
            m_NonemptySegments--;
        }
        if (m_NonemptySegments == 0) {
            DeleteAll();
        }
        m_Mutex->unlock();
    }

    virtual void DeleteAll() {
        if (m_Chunks.empty()) return;
        m_TotalLength = 0;
        m_NonemptySegments = 0;
        m_Chunks.clear();
        for (int i = 0; i < m_Heads.size(); i++) {
            m_Heads[i] = m_Tails[i] = m_ReadPointers[i] = -1;
        }

        m_File->Delete();
    }

private:
    std::unique_ptr<RWFile> m_File;
    int m_NonemptySegments;
    uint64_t m_TotalLength;
    std::vector<Chunk> m_Chunks;
    std::vector<int> m_Heads;
    std::vector<int> m_Tails;
    std::vector<int> m_ReadPointers;
    std::unique_ptr<std::mutex> m_Mutex;
};

class MultiFileStoreImpl : public StoreImpl {
public:
    MultiFileStoreImpl(int maxSegments, const std::string& directory,const std::string& prefix)
        : m_Directory(directory)
        , m_Prefix(prefix + "_")
        , m_Files(maxSegments)
    { 
        if (!m_Directory.ends_with('/')) {
            m_Directory += '/';
        }
        file::CreateDirectory(m_Directory);
    }

    virtual ~MultiFileStoreImpl() noexcept {}

    virtual int MaxSegments() const { return (int)m_Files.size(); }

    virtual bool HasData(int segment) const { return HasFile(segment); }

    virtual uint64_t Length(int segment) const {
        return HasFile(segment) ? GetFile(segment).Length(0) : 0;
    }

    virtual void Rewind(int segment) {
        if (HasFile(segment)) GetFile(segment).Rewind(0);
    }

    virtual void Write(int segment, void* buffer, size_t size) {
        GetFile(segment).Write(0, buffer, size);
    }

    virtual size_t Read(int segment, void* buffer, size_t size) {
        return HasFile(segment) ? GetFile(segment).Read(0, buffer, size) : 0;
    }

    virtual void Delete(int segment) {
        if (HasFile(segment)) {
            m_Files[segment].reset();
        }
    }

private:

    std::string FilePath(int segment) const {
        return m_Directory + m_Prefix + std::to_string(segment);
    }

    bool HasFile(int segment) const { return m_Files[segment].get(); }
    
    StoreImpl& GetFile(int segment) {
        if (!HasFile(segment)) {
            m_Mutex.lock();
            if (!HasFile(segment)) {
                m_Files[segment] = std::make_unique<SingleFileStoreImpl>(1, FilePath(segment));
            }
            m_Mutex.unlock();
        }
        return *m_Files[segment];
    }

    StoreImpl& GetFile(int segment) const {
        return *m_Files[segment];
    }

private:
    std::string m_Directory;
    std::string m_Prefix;
    std::vector<StoreImplRef> m_Files;
    std::mutex m_Mutex;
};

class ParallelStoreImpl : public StoreImpl {
public:
    ParallelStoreImpl(std::vector<StoreImplRef>&& stores)
        : m_Stores(std::move(stores))
    {}

    virtual ~ParallelStoreImpl() noexcept {};

    virtual int MaxSegments() const { return m_Stores[0]->MaxSegments(); }

    virtual bool HasData(int segment) const { return Store(segment).HasData(segment); }

    virtual uint64_t Length(int segment) const { return Store(segment).Length(segment); }
    
    virtual uint64_t TotalLength() const {
        uint64_t total = 0;
        for (const auto& store : m_Stores) {
            total += store->TotalLength();
        }
        return total;
    }

    virtual void Rewind(int segment) { Store(segment).Rewind(segment); }

    virtual void Write(int segment, void* buffer, size_t size) {
        Store(segment).Write(segment, buffer, size);
    }

    virtual size_t Read(int segment, void* buffer, size_t size) {
        return Store(segment).Read(segment, buffer, size);
    }

    virtual void Delete(int segment) { Store(segment).Delete(segment); }

    virtual void DeleteAll() {
        for (auto& store : m_Stores) {
            store->DeleteAll();
        }
    }

private:
    StoreImpl& Store(int segment) { return *m_Stores[segment % m_Stores.size()]; }
    const StoreImpl& Store(int segment) const { return *m_Stores[segment % m_Stores.size()]; }

private:
    std::vector<StoreImplRef> m_Stores;
};

class SequentialStoreImpl : public StoreImpl {
public:
    SequentialStoreImpl(std::vector<StoreImplRef>&& stores)
        : m_Stores(std::move(stores))
    {}

    virtual ~SequentialStoreImpl() noexcept {};

    virtual int MaxSegments() const { return m_Stores[0]->MaxSegments(); }

    virtual bool HasData(int segment) const { return Store(segment).HasData(segment); }

    virtual uint64_t Length(int segment) const { return Store(segment).Length(segment); }

    virtual uint64_t TotalLength() const {
        uint64_t total = 0;
        for (const auto& store : m_Stores) {
            total += store->TotalLength();
        }
        return total;
    }

    virtual void Rewind(int segment) { Store(segment).Rewind(segment); }

    virtual void Write(int segment, void* buffer, size_t size) {
        Store(segment).Write(segment, buffer, size);
    }

    virtual size_t Read(int segment, void* buffer, size_t size) {
        return Store(segment).Read(segment, buffer, size);
    }

    virtual void Delete(int segment) { Store(segment).Delete(segment); }

    virtual void DeleteAll() {
        for (auto& store : m_Stores) {
            store->DeleteAll();
        }
    }

private:
    size_t StoreIdx(int segment) const {
        return segment * m_Stores.size() / MaxSegments();
    }

    StoreImpl& Store(int segment) { return *m_Stores[StoreIdx(segment)]; }
    const StoreImpl& Store(int segment) const { return *m_Stores[StoreIdx(segment)]; }

private:
    std::vector<StoreImplRef> m_Stores;
};

StoreImplRef CreateSingleFileStoreImpl(int maxSegments, const std::string& filePath) {
    return std::make_unique<SingleFileStoreImpl>(maxSegments, filePath);
}

StoreImplRef CreateMultiFileStoreImpl(int maxSegments, const std::string& directory, const std::string& prefix) {
    return std::make_unique<MultiFileStoreImpl>(maxSegments, directory, prefix);
}

StoreImplRef CreateParallelStoreImpl(std::vector<StoreImplRef>&& stores) {
    return std::make_unique<ParallelStoreImpl>(std::move(stores));
}

StoreImplRef CreateSequentialStoreImpl(std::vector<StoreImplRef>&& stores) {
    return std::make_unique<SequentialStoreImpl>(std::move(stores));
}

Store Store::CreateSingleFileStore(int maxSegments, const std::string& filePath) {
    return Store(CreateSingleFileStoreImpl(maxSegments, filePath));
}

Store Store::CreateMultiFileStore(int maxSegments, const std::string& directory, const std::string& prefix) {
    return Store(CreateMultiFileStoreImpl(maxSegments, directory, prefix));
}

Store Store::CreateSequentialStore(int maxSegments, std::vector<std::string> filePaths) {
    std::vector<StoreImplRef> stores;
    for (const auto& filePath : filePaths) {
        stores.emplace_back(CreateSingleFileStoreImpl(maxSegments, filePath));
    }
    return Store(CreateSequentialStoreImpl(std::move(stores)));
}

Store Store::CreateFileStore(int maxSegments, const std::string& name, StoreOptions options) {
    std::vector<StoreImplRef> parallelStores;
    for (const auto& directory: options.directories) {
        file::CreateDirectory(directory);
        if (options.filesPerPath == 0) {
            parallelStores.emplace_back(CreateMultiFileStoreImpl(maxSegments, directory, name));
        }
        else if (options.filesPerPath == 1) {
            parallelStores.emplace_back(CreateSingleFileStoreImpl(maxSegments, directory + "/" + name));
        }
        else {
            std::vector<StoreImplRef> seqStores;
            for (int i = 0; i < options.filesPerPath; i++) {
                std::string file = directory + "/" + name + "_p" + std::to_string(i);
                seqStores.emplace_back(CreateSingleFileStoreImpl(maxSegments, file));
            }
            parallelStores.emplace_back(CreateSequentialStoreImpl(std::move(seqStores)));
        }
    }
    return Store(CreateParallelStoreImpl(std::move(parallelStores)));
}

