#include "BitArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "CompressedFrontier.h"
#include "Multiplexor.h"
#include "Store.h"
#include "ThreadUtil.h"
#include "Util.h"

class DB_Opt3BitBFS_Solver {
public:
    DB_Opt3BitBFS_Solver(
        SegmentedOptions& sopts,
        Store& oldFrontierStore,
        Store& curFrontierStore,
        Store& newFrontierStore,
        StoreSet& curCrossSegmentStores,
        StoreSet& nextCrossSegmentStores)

        : SOpts(sopts)
        , OldFrontierReader(oldFrontierStore)
        , CurFrontierReader(curFrontierStore)
        , FrontierWriter(newFrontierStore)
        , CrossSegmentReader(curCrossSegmentStores)
        , Mult(nextCrossSegmentStores, sopts.Segments)
        , NextArray(SOpts.SegmentSize)
        , CurArray(SOpts.HasOddLengthCycles ? SOpts.SegmentSize : 0)
        , Expander(SOpts.Puzzle)
    { }

    void SetInitialNode(const std::string& initialState) {
        auto initialIndex = SOpts.Puzzle.Parse(initialState);
        auto [seg, idx] = SOpts.GetSegIdx(initialIndex);
        FrontierWriter.SetSegment(seg);
        FrontierWriter.Add(idx);
        FrontierWriter.Flush();

        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [s, idx] = SOpts.GetSegIdx(child);
            if (s == seg) return;
            Mult.Add(op, s, idx);
        };

        Expander.Add(initialIndex, 0, fnExpandCrossSegment);
        Expander.Finish(fnExpandCrossSegment);
        Mult.FlushAllSegments();
    }

    uint64_t Expand(int segment) {
        uint64_t indexBase = (uint64_t)segment << SOpts.Opts.segmentBits;

        bool hasData = false;

        CrossSegmentReader.SetSegment(segment);
        OldFrontierReader.SetSegment(segment);
        CurFrontierReader.SetSegment(segment);
        FrontierWriter.SetSegment(segment);

        for (int op = 0; op < SOpts.OperatorsCount; op++) {
            while (true) {
                auto& vect = CrossSegmentReader.Read(op);
                if (vect.IsEmpty()) break;
                hasData = true;
                for (size_t i = 0; i < vect.Size(); i++) {
                    uint32_t idx = vect[i];
                    NextArray.Set(idx);
                }
            }
        }
        CrossSegmentReader.Delete(segment);

        Expander.SetExpandHint(SOpts.Opts.segmentBits, false);

        auto fnExpandInSegment = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg != segment) return;
            NextArray.Set(idx);
        };

        while (true) {
            auto& vect = CurFrontierReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect[i];
                Expander.Add(indexBase | idx, 0, fnExpandInSegment);
                if (SOpts.HasOddLengthCycles) CurArray.Set(idx);
            }
        }

        if (!hasData) return 0;

        Expander.Finish(fnExpandInSegment);

        while (true) {
            auto& vect = OldFrontierReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect[i];
                NextArray.Clear(idx);
            }
        }

        OldFrontierReader.Delete(segment);

        Expander.SetExpandHint(SOpts.Opts.segmentBits, true);

        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg == segment) return;
            Mult.Add(op, seg, idx);
        };

        uint64_t count = 0;

        if (SOpts.HasOddLengthCycles) {
            NextArray.ScanBitsAndClearWithExcl([&](uint64_t index) {
                count++;
                Expander.Add(indexBase | index, 0, fnExpandCrossSegment);
                FrontierWriter.Add(uint32_t(index));
            }, CurArray);
            CurArray.Clear();
        }
        else {
            NextArray.ScanBitsAndClear([&](uint64_t index) {
                if (SOpts.HasOddLengthCycles && CurArray.Get(index)) return;
                count++;
                Expander.Add(indexBase | index, 0, fnExpandCrossSegment);
                FrontierWriter.Add(uint32_t(index));
            });
        }
        Expander.Finish(fnExpandCrossSegment);
        FrontierWriter.Flush();


        return count;
    }

    void FinishExpand() {
        Mult.FlushAllSegments();
    }

private:
    SegmentedOptions SOpts;
    CompressedSegmentReader OldFrontierReader;
    CompressedSegmentReader CurFrontierReader;
    CompressedSegmentWriter FrontierWriter;
    CompressedCrossSegmentReader CrossSegmentReader;
    CompressedMultiplexor Mult;
    ExpandBuffer Expander;

    BitArray CurArray;
    IndexedBitArray NextArray;
};

std::vector<uint64_t> DiskBasedOptThreeBitBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    Timer timer;
    std::cerr << "DiskBasedOptThreeBitBFS" << std::endl;

    SegmentedOptions sopts(puzzle, opts);

    ensure(opts.segmentBits <= 32);

    sopts.PrintOptions();

    std::vector<uint64_t> result;

    Store oldFrontierStore = sopts.MakeStore("frontier1");
    Store curFrontierStore = sopts.MakeStore("frontier2");
    Store newFrontierStore = sopts.MakeStore("frontier3");
    StoreSet curCrossSegmentStores = sopts.MakeStoreSet("xseg1", sopts.OperatorsCount);
    StoreSet nextCrossSegmentStores = sopts.MakeStoreSet("xseg2", sopts.OperatorsCount);

    auto fnSwapStores = [&]() {
        oldFrontierStore.DeleteAll();
        std::swap(oldFrontierStore, curFrontierStore);
        std::swap(curFrontierStore, newFrontierStore);
        curCrossSegmentStores.Swap(nextCrossSegmentStores);
    };

    std::vector<std::unique_ptr<DB_Opt3BitBFS_Solver>> solvers;
    for (int i = 0; i < opts.threads; i++) {
        solvers.emplace_back(std::make_unique<DB_Opt3BitBFS_Solver>(
            sopts,
            oldFrontierStore,
            curFrontierStore,
            newFrontierStore,
            curCrossSegmentStores,
            nextCrossSegmentStores));
    }

    solvers[0]->SetInitialNode(initialState);
    fnSwapStores();
    result.push_back(1);

    uint64_t total_sz_frontier = 0;
    uint64_t total_sz_xseg = 0;
    uint64_t nanos_collect = 0;
    uint64_t max_size = 0;

    std::cerr << "Step: 0; Count: 1" << std::endl;

    while (result.size() <= opts.maxSteps) {
        Timer stepTimer;

        std::atomic<uint64_t> totalCount{ 0 };

        ParallelExec(opts.threads, sopts.Segments, [&](int thread, int segment) {
            totalCount += solvers[thread]->Expand(segment);
            });

        ParallelExec(opts.threads, [&](int thread) {
            solvers[thread]->FinishExpand();
            });

        if (totalCount == 0) break;
        result.push_back(totalCount);
        fnSwapStores();
        max_size = std::max(max_size, curFrontierStore.TotalLength() + oldFrontierStore.TotalLength() + curCrossSegmentStores.TotalLength());
        std::cerr
            << "Step: " << result.size() - 1
            << "; count: " << WithDecSep(totalCount)
            << " in " << stepTimer
            << "; size: old frontier=" << WithSize(oldFrontierStore.TotalLength())
            << "; size: frontier=" << WithSize(curFrontierStore.TotalLength())
            << ", x-seg=" << WithSize(curCrossSegmentStores.TotalLength())
            << std::endl;
        total_sz_frontier += curFrontierStore.TotalLength();
        total_sz_xseg += curCrossSegmentStores.TotalLength();
    }

    std::cerr << "Time: " << timer << std::endl;
    Store::PrintStats();
    ExpandBuffer::PrintStats();
    StreamVInt::PrintStats();
    std::cerr << "Collect: " << WithTime(nanos_collect) << std::endl;
    std::cerr
        << "Files sizes: frontier=" << WithSize(total_sz_frontier)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    std::cerr << "Max size: " << WithSize(max_size) << std::endl;
    PrintResult(result);
    return result;
}
