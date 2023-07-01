#include "BitArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "CompressedFrontier.h"
#include "Multiplexor.h"
#include "Store.h"
#include "ThreadUtil.h"
#include "Util.h"

template<int BITS>
class DB_OptFS_Solver {
public:
    DB_OptFS_Solver(
        SegmentedOptions& sopts,
        Store& curFrontierStore,
        Store& nextFrontierStore,
        StoreSet& curCrossSegmentStores,
        StoreSet& nextCrossSegmentStores)

        : SOpts(sopts)
        , FrontierReader(curFrontierStore)
        , FrontierWriter(nextFrontierStore)
        , CrossSegmentReader(curCrossSegmentStores)
        //, Mult(nextCrossSegmentStores, sopts.Segments)
        , Mult(nextCrossSegmentStores, sopts.OperatorsCount, sopts.Segments, 16, 12, 16ui64 * 1024 * 1024)
        , Array(SOpts.SegmentSize)
        , FrontierArray(SOpts.HasOddLengthCycles ? SOpts.SegmentSize : 0)
        , Expander(SOpts.Puzzle)
    { }

    void SetInitialNode(const std::string& initialState) {
        auto initialIndex = SOpts.Puzzle.Parse(initialState);
        auto [seg, idx] = SOpts.GetSegIdx(initialIndex);
        FrontierWriter.SetSegment(seg);
        FrontierWriter.Add(idx, 0);
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
        Timer timerExpand;
        uint64_t indexBase = (uint64_t)segment << SOpts.Opts.segmentBits;

        bool hasData = false;

        CrossSegmentReader.SetSegment(segment);
        FrontierReader.SetSegment(segment);
        FrontierWriter.SetSegment(segment);

        for (int op = 0; op < SOpts.OperatorsCount; op++) {
            while (true) {
                auto& vect = CrossSegmentReader.Read(op);
                if (vect.IsEmpty()) break;
                hasData = true;
                for (size_t i = 0; i < vect.Size(); i++) {
                    uint32_t idx = vect[i];
                    Array.Set(idx, op);
                }
            }
        }
        CrossSegmentReader.Delete(segment);

        Expander.SetExpandHint(SOpts.Opts.segmentBits, false);

        auto fnExpandInSegment = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg != segment) return;
            Array.Set(idx, op);
        };

        while (true) {
            auto vect = FrontierReader.Read();
            if (vect.IsEmpty()) break;
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect.Indexes[i];
                uint8_t opBits = vect.OpBits[i];
                Expander.Add(indexBase | idx, opBits, fnExpandInSegment);
                if (SOpts.HasOddLengthCycles) FrontierArray.Set(idx);
            }
        }

        if (!hasData) return 0;

        Expander.Finish(fnExpandInSegment);

        FrontierReader.Delete(segment);

        m_NanosExpand += timerExpand.Elapsed();
        Timer timerCollect;

        Expander.SetExpandHint(SOpts.Opts.segmentBits, true);

        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg == segment) return;
            Mult.Add(op, seg, idx);
        };

        uint64_t count = 0;

        Array.ScanBitsAndClear([&](uint64_t index, int opBits) {
            if (SOpts.HasOddLengthCycles && FrontierArray.Get(index)) return;
            count++;
            if (opBits < SOpts.OperatorsMask) {
                Expander.Add(indexBase | index, opBits, fnExpandCrossSegment);
                FrontierWriter.Add(uint32_t(index), opBits);
            }
        });
        Expander.Finish(fnExpandCrossSegment);
        FrontierWriter.Flush();

        if (SOpts.HasOddLengthCycles) FrontierArray.Clear();

        m_NanosCollect += timerCollect.Elapsed();

        return count;
    }

    void FinishExpand() {
        Mult.FlushAllSegments();
    }

    static void PrintStats() {
        std::cerr << "Expand: " << WithTime(m_NanosExpand) <<
            "; Collect: " << WithTime(m_NanosCollect) << std::endl;
    }
private:
    SegmentedOptions SOpts;
    CompressedFrontierReader FrontierReader;
    CompressedFrontierWriter FrontierWriter;
    CompressedCrossSegmentReader CrossSegmentReader;
    //CompressedMultiplexor Mult;
    SmartMultiplexor Mult;
    ExpandBuffer Expander;

    //MultiBitArray<BITS> Array;
    IndexedArray<BITS> Array;
    BitArray FrontierArray;

private:
    static std::atomic<uint64_t> m_NanosExpand;
    static std::atomic<uint64_t> m_NanosCollect;
};

template<int BITS>
std::atomic<uint64_t> DB_OptFS_Solver<BITS>::m_NanosExpand{ 0 };
template<int BITS>
std::atomic<uint64_t> DB_OptFS_Solver<BITS>::m_NanosCollect{ 0 };


template<int BITS>
std::vector<uint64_t> DiskBasedOptFrontierSearchInt(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    Timer timer;
    std::cerr << "DiskBasedFrontierSearch2" << std::endl;

    SegmentedOptions sopts(puzzle, opts);

    ensure(opts.segmentBits <= 32);
    ensure(sopts.OperatorsCount <= 8);

    sopts.PrintOptions();

    std::vector<uint64_t> result;

    Store curFrontierStore = sopts.MakeStore("frontier1");
    Store newFrontierStore = sopts.MakeStore("frontier2");
    StoreSet curCrossSegmentStores = sopts.MakeStoreSet("xseg1", sopts.OperatorsCount);
    StoreSet nextCrossSegmentStores = sopts.MakeStoreSet("xseg2", sopts.OperatorsCount);

    auto fnSwapStores = [&]() {
        curFrontierStore.DeleteAll();
        std::swap(curFrontierStore, newFrontierStore);
        curCrossSegmentStores.Swap(nextCrossSegmentStores);
    };

    std::vector<std::unique_ptr<DB_OptFS_Solver<BITS>>> solvers;
    for (int i = 0; i < opts.threads; i++) {
        solvers.emplace_back(std::make_unique<DB_OptFS_Solver<BITS>>(
            sopts,
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
        max_size = std::max(max_size, curFrontierStore.TotalLength() + curCrossSegmentStores.TotalLength());
        std::cerr
            << "Step: " << result.size() - 1
            << "; count: " << WithDecSep(totalCount)
            << " in " << stepTimer
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
    DB_OptFS_Solver<BITS>::PrintStats();
    std::cerr << "Collect: " << WithTime(nanos_collect) << std::endl;
    std::cerr
        << "Files sizes: frontier=" << WithSize(total_sz_frontier)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    std::cerr << "Max size: " << WithSize(max_size) << std::endl;
    PrintResult(result);
    return result;
}

std::vector<uint64_t> DiskBasedOptFrontierSearch(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    int bits = puzzle.OperatorsCount();
    ensure(bits > 0 && bits <= 8);
    // TODO: op bits compression is currently supported for 4-bit values
    ensure(bits <= 4);
    if (bits == 1)
        return DiskBasedOptFrontierSearchInt<1>(puzzle, initialState, opts);
    else if (bits == 2)
        return DiskBasedOptFrontierSearchInt<2>(puzzle, initialState, opts);
    else if (bits <= 4)
        return DiskBasedOptFrontierSearchInt<4>(puzzle, initialState, opts);
    else
        return DiskBasedOptFrontierSearchInt<8>(puzzle, initialState, opts);
}
