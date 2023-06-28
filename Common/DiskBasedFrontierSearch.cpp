#include "BitArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
#include "ThreadUtil.h"
#include "Util.h"

#include <thread>

template<int BITS>
class DB_FS_Solver {
public:
    DB_FS_Solver(
        SegmentedOptions& sopts,
        Store& curFrontierStore,
        Store& nextFrontierStore,
        Store& crossSegmentStore)

        : SOpts(sopts)
        , FrontierReader(curFrontierStore)
        , FrontierWriter(nextFrontierStore)
        , CrossSegmentReader(crossSegmentStore)
        , Mult(crossSegmentStore, SOpts.Segments)
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
    }

    void Expand(int segment) {
        uint64_t indexBase = (uint64_t)segment << SOpts.Opts.segmentBits;

        FrontierReader.SetSegment(segment);

        auto fnExpand = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            Mult.Add(seg, GetValue(idx, op));
        };

        while (true) {
            auto& vect = FrontierReader.Read();
            if (vect.IsEmpty()) break;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t val = vect[i];
                auto [idx, op] = GetIndexAndOp(val);
                Expander.Add(indexBase | idx, op, fnExpand);
            }
        }
        Expander.Finish(fnExpand);

        if (!SOpts.HasOddLengthCycles) FrontierReader.Delete(segment);
    }

    void FinishExpand() {
        Mult.FlushAllSegments();
    }

    uint64_t Collect(int segment) {
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
        CrossSegmentReader.Delete(segment);

        if (!hasData) {
            if (SOpts.HasOddLengthCycles) {
                FrontierReader.Delete(segment);
            }
            return 0;
        }

        if (SOpts.HasOddLengthCycles) {
            while (true) {
                auto& vect = FrontierReader.Read();
                if (vect.IsEmpty()) break;
                for (size_t i = 0; i < vect.Size(); i++) {
                    uint32_t val = vect[i];
                    auto [idx, op] = GetIndexAndOp(val);
                    FrontierArray.Set(idx);
                }
            }
            FrontierReader.Delete(segment);
        }

        uint64_t count = 0;
        Array.ScanBitsAndClear([&](uint64_t index, int opBits) {
            if (SOpts.HasOddLengthCycles && FrontierArray.Get(index)) return;
            count++;
            if (opBits < SOpts.OperatorsMask) {
                FrontierWriter.Add(GetValue(uint32_t(index), opBits));
            }
        });
        FrontierWriter.Flush();

        if (SOpts.HasOddLengthCycles) {
            FrontierArray.Clear();
        }

        return count;
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
    SegmentReader FrontierReader;
    SegmentWriter FrontierWriter;
    SegmentReader CrossSegmentReader;
    SimpleMultiplexor Mult;
    ExpandBuffer Expander;

    IndexedArray<BITS> Array;
    BitArray FrontierArray;
};

template <int BITS>
std::vector<uint64_t> DiskBasedFrontierSearchInternal(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    Timer timer;
    SegmentedOptions sopts(puzzle, opts);

    // classic FrontierSearch stores indexes and used operator bits in a single 4-byte word
    ensure(opts.segmentBits + sopts.OperatorsCount <= 32);

    sopts.PrintOptions();

    std::vector<uint64_t> result;

    Store curFrontierStore = sopts.MakeStore("frontier1");
    Store newFrontierStore = sopts.MakeStore("frontier2");
    Store crossSegmentStore = sopts.MakeStore("xseg");

    std::vector<std::unique_ptr<DB_FS_Solver<BITS>>> solvers;
    for (int i = 0; i < opts.threads; i++) {
        solvers.emplace_back(std::make_unique<DB_FS_Solver<BITS>>(
            sopts,
            curFrontierStore,
            newFrontierStore,
            crossSegmentStore
        ));
    }

    solvers[0]->SetInitialNode(initialState);
    std::swap(curFrontierStore, newFrontierStore);
    result.push_back(1);

    uint64_t total_sz_frontier = 0;
    uint64_t total_sz_xseg = 0;
    uint64_t nanos_collect = 0;

    std::cerr << "Step: 0; Count: 1" << std::endl;

    while (result.size() <= opts.maxSteps) {
        Timer stepTimer;
        ParallelExec(opts.threads, sopts.Segments, [&](int thread, int segment) {
            solvers[thread]->Expand(segment);
        });

        ParallelExec(opts.threads, [&](int thread) {
            solvers[thread]->FinishExpand();
        });

        uint64_t xstore_size = crossSegmentStore.TotalLength();

        std::atomic<uint64_t> totalCount{ 0 };

        ParallelExec(opts.threads, sopts.Segments, [&](int thread, int segment) {
            totalCount += solvers[thread]->Collect(segment);
        });

        if (totalCount == 0) break;
        result.push_back(totalCount);
        curFrontierStore.DeleteAll();
        crossSegmentStore.DeleteAll();
        std::swap(curFrontierStore, newFrontierStore);
        std::cerr
            << "Step: " << result.size() - 1
            << "; count: " << totalCount
            << " in " << stepTimer
            << "; size: frontier=" << WithSize(curFrontierStore.TotalLength())
            << ", x-seg=" << WithSize(xstore_size)
            << std::endl;
        total_sz_frontier += curFrontierStore.TotalLength();
        total_sz_xseg += xstore_size;
    }

    std::cerr << "Time: " << timer << std::endl;
    Store::PrintStats();
    ExpandBuffer::PrintStats();
    //std::cerr << "Collect: " << WithTime(nanos_collect) << std::endl;
    std::cerr
        << "Total files: frontier=" << WithSize(total_sz_frontier)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    return result;
}

std::vector<uint64_t> DiskBasedFrontierSearch(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    int bits = puzzle.OperatorsCount();
    ensure(bits > 0 && bits <= 16);
    if (bits == 1) 
        return DiskBasedFrontierSearchInternal<1>(puzzle, initialState, opts);
    else if (bits == 2)
        return DiskBasedFrontierSearchInternal<2>(puzzle, initialState, opts);
    else if (bits <= 4)
        return DiskBasedFrontierSearchInternal<4>(puzzle, initialState, opts);
    else if (bits <= 8)
        return DiskBasedFrontierSearchInternal<8>(puzzle, initialState, opts);
    else
        return DiskBasedFrontierSearchInternal<16>(puzzle, initialState, opts);
}
