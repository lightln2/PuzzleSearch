#include "BitArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
#include "ThreadUtil.h"
#include "Util.h"

#include <thread>

namespace {

    void LoadBitArray(int segment, Store& store, BitArray& arr) {
        arr.Clear();
        auto read = store.ReadArray(segment, arr.Data(), arr.DataSize());
        ensure(read == 0 || read == arr.DataSize());
        store.Delete(segment);
    };

    void SaveBitArray(int segment, Store& store, BitArray& arr) {
        store.WriteArray(segment, arr.Data(), arr.DataSize());
        arr.Clear();
    };


} // namespace

class DB_BFS_Solver {
public:
    DB_BFS_Solver(
        SegmentedOptions sopts,
        Store& currentOpenListStore,
        Store& nextOpenListStore,
        Store& currentClosedListStore,
        Store& nextClosedListStore,
        Store& currentCrossSegmentStore,
        Store& nextCrossSegmentStore)
        
        : SOpts(sopts)
        , CurrentOpenListStore(currentOpenListStore)
        , NextOpenListStore(nextOpenListStore)
        , CurrentClosedListStore(currentClosedListStore)
        , NextClosedListStore(nextClosedListStore)
        , CurrentCrossSegmentStore(currentCrossSegmentStore)
        , NextCrossSegmentStore(nextCrossSegmentStore)
        , CurrentXSegReader(currentCrossSegmentStore)
        , Mult(nextCrossSegmentStore, SOpts.Segments)
        , Expander(SOpts.Puzzle)
        , ClosedList(SOpts.SegmentSize)
        , OpenList(SOpts.SegmentSize)
        , NewOpenList(SOpts.SegmentSize)
    {
    }

    void SetInitialNode(const std::string& initialState) {
        auto initialIndex = SOpts.Puzzle.Parse(initialState);
        auto [seg, idx] = SOpts.GetSegIdx(initialIndex);
        NewOpenList.Set(idx);
        SaveOpenList(seg);
    }

    uint64_t Process(int segment) {
        uint64_t indexBase = (uint64_t)segment << SOpts.Opts.segmentBits;
        LoadOpenList(segment);
        LoadClosedList(segment);
        CurrentXSegReader.SetSegment(segment);

        while (true) {
            auto& vect = CurrentXSegReader.Read();
            if (vect.IsEmpty()) break;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect[i];
                OpenList.Set(idx);
            }
        }
        CurrentXSegReader.Delete(segment);

        uint64_t totalCount = OpenList.AndNotAndCount(ClosedList);
        ClosedList.Or(OpenList);

        auto fnExpand = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg == segment) NewOpenList.Set(idx);
            else Mult.Add(seg, idx);
        };

        OpenList.ScanBitsAndClear([&](uint64_t index) {
            Expander.Add(indexBase | index, 0, fnExpand);
        });
        Expander.Finish(fnExpand);

        NewOpenList.AndNot(ClosedList);

        SaveOpenList(segment);
        SaveClosedList(segment);

        return totalCount;
    }

    void FinishProcess() {
        Mult.FlushAllSegments();
    }

private:
    void LoadClosedList(int segment) {
        LoadBitArray(segment, CurrentClosedListStore, ClosedList);
    }

    void SaveClosedList(int segment) {
        SaveBitArray(segment, NextClosedListStore, ClosedList);
    };

    void LoadOpenList(int segment) {
        LoadBitArray(segment, CurrentOpenListStore, OpenList);
    };

    void SaveOpenList(int segment) {
        SaveBitArray(segment, NextOpenListStore, NewOpenList);
    };

private:
    SegmentedOptions SOpts;
    Store& CurrentOpenListStore;
    Store& NextOpenListStore;
    Store& CurrentClosedListStore;
    Store& NextClosedListStore;
    Store& CurrentCrossSegmentStore;
    Store& NextCrossSegmentStore;
    SegmentReader CurrentXSegReader;
    SimpleMultiplexor Mult;
    ExpandBuffer Expander;

    BitArray ClosedList;
    BitArray OpenList;
    BitArray NewOpenList;
};


std::vector<uint64_t> DiskBasedClassicBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    ensure(opts.segmentBits <= 32);
    Timer timer;
    SegmentedOptions sopts(puzzle, opts);
    sopts.PrintOptions();

    std::vector<uint64_t> result;

    Store currentOpenListStore = sopts.MakeStore("open1");
    Store nextOpenListStore = sopts.MakeStore("open2");
    Store currentClosedListStore = sopts.MakeStore("closed1");
    Store nextClosedListStore = sopts.MakeStore("closed2");
    Store currentCrossSegmentStore = sopts.MakeStore("xseg1");
    Store nextCrossSegmentStore = sopts.MakeStore("xseg2");

    auto fnSwapStores = [&]() {
        currentOpenListStore.DeleteAll();
        currentClosedListStore.DeleteAll();
        currentCrossSegmentStore.DeleteAll();
        std::swap(currentOpenListStore, nextOpenListStore);
        std::swap(currentClosedListStore, nextClosedListStore);
        std::swap(currentCrossSegmentStore, nextCrossSegmentStore);
    };

    std::vector<std::unique_ptr<DB_BFS_Solver>> solvers;
    for (int i = 0; i < opts.threads; i++) {
        solvers.emplace_back(std::make_unique<DB_BFS_Solver>(
            sopts,
            currentOpenListStore,
            nextOpenListStore,
            currentClosedListStore,
            nextClosedListStore,
            currentCrossSegmentStore,
            nextCrossSegmentStore));
    }
    
    solvers[0]->SetInitialNode(initialState);
    fnSwapStores();

    uint64_t total_sz_open = 0;
    uint64_t total_sz_closed = 0;
    uint64_t total_sz_xseg = 0;

    while (true) {
        Timer stepTimer;

        std::atomic<uint64_t> totalCount{ 0 };

        ParallelExec(opts.threads, sopts.Segments, [&](int thread, int segment) {
            totalCount += solvers[thread]->Process(segment);
        });

        ParallelExec(opts.threads, [&](int thread) {
            solvers[thread]->FinishProcess();
        });

        if (totalCount == 0) break;
        result.push_back(totalCount);
        fnSwapStores();
        std::cerr 
            << "Step: " << result.size() - 1
            << "; count: " << totalCount
            << " in " << stepTimer 
            << "; size: open=" << WithSize(currentOpenListStore.TotalLength())
            << ", closed=" << WithSize(currentClosedListStore.TotalLength())
            << ", x-seg=" << WithSize(currentCrossSegmentStore.TotalLength())
            << std::endl;
        total_sz_open += currentOpenListStore.TotalLength();
        total_sz_closed += currentClosedListStore.TotalLength();
        total_sz_xseg += currentCrossSegmentStore.TotalLength();
    }

    std::cerr << "Time: " << timer << std::endl;
    Store::PrintStats();
    ExpandBuffer::PrintStats();
    std::cerr
        << "Total files: open=" << WithSize(total_sz_open)
        << "; closed=" << WithSize(total_sz_closed)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    return result;
}
