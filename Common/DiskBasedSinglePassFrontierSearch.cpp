#include "BitArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
#include "ThreadUtil.h"
#include "Util.h"

template<int BITS>
class DB_SPFS_Solver {
public:
    DB_SPFS_Solver(
        SegmentedOptions& sopts,
        Store& curFrontierStore,
        Store& nextFrontierStore,
        Store& curCrossSegmentStore,
        Store& nextCrossSegmentStore)

        : SOpts(sopts)
        , CurFrontierStore(curFrontierStore)
        , NextFrontierStore(nextFrontierStore)
        , CurCrossSegmentStore(curCrossSegmentStore)
        , NextCrossSegmentStore(nextCrossSegmentStore)
        , FrontierReader(curFrontierStore)
        , FrontierWriter(nextFrontierStore)
        , CrossSegmentReader(curCrossSegmentStore)
        , Mult(nextCrossSegmentStore, SOpts.Segments)
        , Expander(SOpts.Puzzle)
        , Array(SOpts.SegmentSize)
        , FrontierArray(SOpts.SegmentSize)
    {
    }

    void SetInitialNode(const std::string& initialState) {
        auto initialIndex = SOpts.Puzzle.Parse(initialState);
        auto [seg, idx] = SOpts.GetSegIdx(initialIndex);
        FrontierWriter.SetSegment(seg);
        FrontierWriter.Add(GetValue(idx, 0));
        FrontierWriter.Flush();

        //Expand cross-segment
        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [s, idx] = SOpts.GetSegIdx(child);
            if (s == seg) return;
            Mult.Add(s, GetValue(idx, op));
        };

        Expander.Add(initialIndex, 0, fnExpandCrossSegment);
        Expander.Finish(fnExpandCrossSegment);
        Mult.FlushAllSegments();
        std::swap(CurFrontierStore, NextFrontierStore);
        std::swap(CurCrossSegmentStore, NextCrossSegmentStore);
    }

    uint64_t Expand(int segment) {
        uint64_t indexBase = (uint64_t)segment << SOpts.Opts.segmentBits;

        bool hasData = false;

        CrossSegmentReader.SetSegment(segment);
        FrontierReader.SetSegment(segment);
        FrontierWriter.SetSegment(segment);

        while (true) {
            auto& vect = CrossSegmentReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t val = vect[i];
                auto [idx, op] = GetIndexAndOp(val);
                Array.Set(idx, op);
            }
        }
        CurCrossSegmentStore.Delete(segment);

        auto fnExpandInSegment = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg != segment) return;
            Array.Set(idx, op);
        };

        while (true) {
            auto& vect = FrontierReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t val = vect[i];
                auto [idx, op] = GetIndexAndOp(val);
                Expander.Add(indexBase | idx, op, fnExpandInSegment);
                if (SOpts.Puzzle.HasOddLengthCycles()) FrontierArray.Set(idx);
            }
        }

        if (!hasData) return 0;

        Expander.Finish(fnExpandInSegment);
        CurFrontierStore.Delete(segment);

        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg == segment) return;
            Mult.Add(seg, GetValue(idx, op));
        };

        uint64_t count = 0;

        Array.ScanBitsAndClear([&](uint64_t index, int opBits) {
            if (SOpts.Puzzle.HasOddLengthCycles() && FrontierArray.Get(index)) return;
            count++;
            if (opBits < SOpts.OperatorsMask) {
                Expander.Add(indexBase | index, opBits, fnExpandCrossSegment);
                FrontierWriter.Add(GetValue(uint32_t(index), opBits));
            }
        });
        Expander.Finish(fnExpandCrossSegment);
        FrontierWriter.Flush();

        if (SOpts.Puzzle.HasOddLengthCycles()) FrontierArray.Clear();

        return count;
    }

    void FinishExpand() {
        Mult.FlushAllSegments();
    }

private:
    std::pair<uint32_t, int> GetIndexAndOp(uint32_t value) {
        return { value >> SOpts.OperatorsCount, value & SOpts.OperatorsMask };
    };

    uint32_t GetValue(uint32_t index, int op) {
        return (index << SOpts.OperatorsCount) | op;
    };

private:
    SegmentedOptions SOpts;
    Store& CurFrontierStore;
    Store& NextFrontierStore;
    Store& CurCrossSegmentStore;
    Store& NextCrossSegmentStore;
    SegmentReader FrontierReader;
    SegmentWriter FrontierWriter;
    SegmentReader CrossSegmentReader;
    Multiplexor Mult;
    ExpandBuffer Expander;

    MultiBitArray<BITS> Array;
    BitArray FrontierArray;
};

template<int BITS>
std::vector<uint64_t> DiskBasedSinglePassFrontierSearchInt(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    Timer timer;
    std::cerr << "DiskBasedSinglePassFrontierSearch" << std::endl;

    SegmentedOptions sopts(puzzle, opts);

    // classic FrontierSearch stores indexes and used operator bits in a single 4-byte word
    ensure(opts.segmentBits + sopts.OperatorsCount <= 32);

    sopts.PrintOptions();

    std::vector<uint64_t> result;

    Store curFrontierStore = sopts.MakeStore("frontier1");
    Store newFrontierStore = sopts.MakeStore("frontier2");
    Store curCrossSegmentStore = sopts.MakeStore("xseg1");
    Store nextCrossSegmentStore = sopts.MakeStore("xseg2");

    std::vector<std::unique_ptr<DB_SPFS_Solver<BITS>>> solvers;
    for (int i = 0; i < opts.threads; i++) {
        solvers.emplace_back(std::make_unique<DB_SPFS_Solver<BITS>>(
            sopts,
            curFrontierStore,
            newFrontierStore,
            curCrossSegmentStore,
            nextCrossSegmentStore));
    }

    solvers[0]->SetInitialNode(initialState);
    result.push_back(1);

    uint64_t total_sz_frontier = 0;
    uint64_t total_sz_xseg = 0;
    uint64_t nanos_collect = 0;

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
        curFrontierStore.DeleteAll();
        curCrossSegmentStore.DeleteAll();
        std::swap(curFrontierStore, newFrontierStore);
        std::swap(curCrossSegmentStore, nextCrossSegmentStore);
        std::cerr
            << "Step: " << result.size() - 1
            << "; count: " << WithDecSep(totalCount)
            << " in " << stepTimer
            << "; size: frontier=" << WithSize(curFrontierStore.TotalLength())
            << ", x-seg=" << WithSize(curCrossSegmentStore.TotalLength())
            << std::endl;
        total_sz_frontier += curFrontierStore.TotalLength();
        total_sz_xseg += curCrossSegmentStore.TotalLength();
    }

    std::cerr << "Time: " << timer << std::endl;
    Store::PrintStats();
    ExpandBuffer::PrintStats();
    std::cerr << "Collect: " << WithTime(nanos_collect) << std::endl;
    std::cerr
        << "Total files: frontier=" << WithSize(total_sz_frontier)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    return result;
}

std::vector<uint64_t> DiskBasedSinglePassFrontierSearch(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    int bits = puzzle.OperatorsCount();
    ensure(bits > 0 && bits <= 16);
    if (bits == 1)
        return DiskBasedSinglePassFrontierSearchInt<1>(puzzle, initialState, opts);
    else if (bits == 2)
        return DiskBasedSinglePassFrontierSearchInt<2>(puzzle, initialState, opts);
    else if (bits <= 4)
        return DiskBasedSinglePassFrontierSearchInt<4>(puzzle, initialState, opts);
    else if (bits <= 8)
        return DiskBasedSinglePassFrontierSearchInt<8>(puzzle, initialState, opts);
    else
        return DiskBasedSinglePassFrontierSearchInt<16>(puzzle, initialState, opts);
}
