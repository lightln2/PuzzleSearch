#include "BitArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
#include "ThreadUtil.h"
#include "Util.h"

namespace {

    void LoadBitArray(int segment, Store& store, BitArray& arr) {
        arr.Clear();
        store.Rewind(segment);
        auto read = store.ReadArray(segment, arr.Data(), arr.DataSize());
        ensure(read == 0 || read == arr.DataSize());
    };

    void SaveBitArray(int segment, Store& store, BitArray& arr) {
        store.WriteArray(segment, arr.Data(), arr.DataSize());
        arr.Clear();
    };


} // namespace

class DB_3BitBFS_Solver {
public:
    DB_3BitBFS_Solver(
        SegmentedOptions sopts,
        Store& oldStore,
        Store& curStore,
        Store& newStore,
        Store& curCrossSegmentStore,
        Store& nextCrossSegmentStore)

        : SOpts(sopts)
        , OldStore(oldStore)
        , CurStore(curStore)
        , NewStore(newStore)
        , CurCrossSegmentStore(curCrossSegmentStore)
        , NextCrossSegmentStore(nextCrossSegmentStore)
        , CrossSegmentReader(curCrossSegmentStore)
        , Mult(nextCrossSegmentStore, SOpts.Segments)
        , Expander(SOpts.Puzzle)
        , OldList(SOpts.SegmentSize)
        , CurList(SOpts.SegmentSize)
        , NewList(SOpts.SegmentSize)
    {
    }

    void SetInitialNode(const std::string& initialState) {
        auto initialIndex = SOpts.Puzzle.Parse(initialState);
        auto [seg, idx] = SOpts.GetSegIdx(initialIndex);
        NewList.Set(idx);
        Save(seg);
        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [s, idx] = SOpts.GetSegIdx(child);
            if (s == seg) return;
            Mult.Add(s, idx);
        };
        Expander.Add(initialIndex, 0, fnExpandCrossSegment);
        Expander.Finish(fnExpandCrossSegment);
        Mult.FlushAllSegments();
        std::swap(OldStore, CurStore);
        std::swap(CurStore, NewStore);
        std::swap(CurCrossSegmentStore, NextCrossSegmentStore);
    }

    uint64_t Process(int segment) {
        uint64_t totalCount = 0;

        uint64_t indexBase = (uint64_t)segment << SOpts.Opts.segmentBits;

        LoadOld(segment);
        LoadCur(segment);
        CrossSegmentReader.SetSegment(segment);

        while (true) {
            auto& vect = CrossSegmentReader.Read();
            if (vect.IsEmpty()) break;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect[i];
                NewList.Set(idx);
            }
        }
        CurCrossSegmentStore.Delete(segment);

        auto fnExpandInSegment = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg != segment) return;
            NewList.Set(idx);
        };

        CurList.ScanBits([&](uint64_t index) {
            Expander.Add(indexBase | index, 0, fnExpandInSegment);
        });
        Expander.Finish(fnExpandInSegment);

        NewList.AndNot(CurList);
        NewList.AndNot(OldList);

        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg == segment) return;
            Mult.Add(seg, idx);
        };
        NewList.ScanBits([&](uint64_t index) {
            Expander.Add(indexBase | index, 0, fnExpandCrossSegment);
        });
        Expander.Finish(fnExpandCrossSegment);

        uint64_t count = NewList.BitsCount();
        Save(segment);

        return count;
    }

    void FinishProcess() {
        Mult.FlushAllSegments();
    }

private:
    void LoadOld(int segment) {
        LoadBitArray(segment, OldStore, OldList);
        OldStore.Delete(segment);
    };
    void LoadCur(int segment) {
        LoadBitArray(segment, CurStore, CurList);
    };
    void Save(int segment) {
        SaveBitArray(segment, NewStore, NewList);
    };

private:
    SegmentedOptions SOpts;
    Store& OldStore;
    Store& CurStore;
    Store& NewStore;
    Store& CurCrossSegmentStore;
    Store& NextCrossSegmentStore;
    SegmentReader CrossSegmentReader;
    SimpleMultiplexor Mult;
    ExpandBuffer Expander;

    BitArray OldList;
    BitArray CurList;
    BitArray NewList;
};


std::vector<uint64_t> DiskBasedThreeBitBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    std::cerr << "DiskBased ThreeBit BFS" << std::endl;

    ensure(opts.segmentBits <= 32);
    Timer timer;
    SegmentedOptions sopts(puzzle, opts);
    sopts.PrintOptions();

    std::vector<uint64_t> result;

    Store oldStore = sopts.MakeStore("store1");
    Store curStore = sopts.MakeStore("store2");
    Store newStore = sopts.MakeStore("store3");
    Store curCrossSegmentStore = sopts.MakeStore("xseg1");
    Store nextCrossSegmentStore = sopts.MakeStore("xseg2");

    std::vector<std::unique_ptr<DB_3BitBFS_Solver>> solvers;
    for (int i = 0; i < opts.threads; i++) {
        solvers.emplace_back(std::make_unique<DB_3BitBFS_Solver>(
            sopts,
            oldStore,
            curStore,
            newStore,
            curCrossSegmentStore,
            nextCrossSegmentStore));
    }

    solvers[0]->SetInitialNode(initialState);

    result.push_back(1);

    uint64_t total_sz_old = 0;
    uint64_t total_sz_cur = 0;
    uint64_t total_sz_xseg = 0;

    std::cerr << "Step: 1; Count: 1" << std::endl;

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
        oldStore.DeleteAll();
        curCrossSegmentStore.DeleteAll();
        std::swap(oldStore, curStore);
        std::swap(curStore, newStore);
        std::swap(curCrossSegmentStore, nextCrossSegmentStore);
        std::cerr
            << "Step: " << result.size()
            << "; count: " << WithDecSep(totalCount)
            << " in " << stepTimer
            << "; size: old=" << WithSize(oldStore.TotalLength())
            << ", cur=" << WithSize(curStore.TotalLength())
            << ", x-seg=" << WithSize(curCrossSegmentStore.TotalLength())
            << std::endl;
        total_sz_old += oldStore.TotalLength();
        total_sz_cur += curStore.TotalLength();
        total_sz_xseg += curCrossSegmentStore.TotalLength();
    }

    std::cerr << "Time: " << timer << std::endl;
    Store::PrintStats();
    std::cerr
        << "Total files: old=" << WithSize(total_sz_old)
        << "; cur=" << WithSize(total_sz_cur)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    return result;
}
