#include "BoolArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
#include "Util.h"

#include <thread>

class DB_FS_Solver {
public:
    DB_FS_Solver(
        Puzzle& puzzle,
        PuzzleOptions& opts,
        int segments,
        uint64_t segmentSize,
        uint64_t segmentMask,
        int operatorsCount,
        int operatorsMask,
        Store& curFrontierStore,
        Store& nextFrontierStore,
        Store& crossSegmentStore)

        : Puzzle(puzzle)
        , Opts(opts)
        , Segments(segments)
        , SegmentSize(segmentSize)
        , SegmentMask(segmentMask)
        , OperatorsCount(operatorsCount)
        , OperatorsMask(operatorsMask)
        , CurFrontierStore(curFrontierStore)
        , NextFrontierStore(nextFrontierStore)
        , CrossSegmentStore(crossSegmentStore)
        , FrontierReader(curFrontierStore)
        , FrontierWriter(nextFrontierStore)
        , CrossSegmentReader(crossSegmentStore)
        , Mult(crossSegmentStore, Segments)
        , Expander(puzzle)
        , Array(OperatorsCount, SegmentSize)
        , FrontierArray(SegmentSize)
    {
    }

    void SetInitialNode(const std::string& initialState) {
        auto initialIndex = Puzzle.Parse(initialState);
        auto [seg, idx] = GetSegIdx(initialIndex);
        FrontierWriter.SetSegment(seg);
        FrontierWriter.Add(GetValue(idx, 0));
        FrontierWriter.Flush();
        std::swap(CurFrontierStore, NextFrontierStore);
    }

    void Expand(int segment) {
        uint64_t indexBase = (uint64_t)segment << Opts.segmentBits;

        FrontierReader.SetSegment(segment);

        auto fnExpand = [&](uint64_t child, int op) {
            auto [seg, idx] = GetSegIdx(child);
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

        if (!Puzzle.HasOddLengthCycles()) CurFrontierStore.Delete(segment);
    }

    void FinishExpand() {
        Mult.FlushAllSegments();
    }

    uint64_t Collect(int segment) {
        uint64_t indexBase = (uint64_t)segment << Opts.segmentBits;

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
        CrossSegmentStore.Delete(segment);

        if (Puzzle.HasOddLengthCycles()) {
            while (true) {
                auto& vect = FrontierReader.Read();
                if (vect.IsEmpty()) break;
                for (size_t i = 0; i < vect.Size(); i++) {
                    uint32_t val = vect[i];
                    auto [idx, op] = GetIndexAndOp(val);
                    FrontierArray.Set(idx);
                }
            }
            CurFrontierStore.Delete(segment);
        }

        if (!hasData) return 0;

        uint64_t count = 0;
        Array.ScanBitsAndClear([&](uint64_t index, int opBits) {
            if (Puzzle.HasOddLengthCycles() && FrontierArray.Get(index)) return;
            count++;
            if (opBits < OperatorsMask) {
                FrontierWriter.Add(GetValue(uint32_t(index), opBits));
            }
        });
        FrontierWriter.Flush();

        if (Puzzle.HasOddLengthCycles()) {
            FrontierArray.Clear();
        }

        return count;
    }

private:
    std::pair<int, uint32_t> GetSegIdx(uint64_t index) {
        return { int(index >> Opts.segmentBits), uint32_t(index & SegmentMask) };
    };

    std::pair<uint32_t, int> GetIndexAndOp(uint32_t value) {
        return { value >> OperatorsCount, value & OperatorsMask };
    };

    uint32_t GetValue(uint32_t index, int op) {
        return (index << OperatorsCount) | op;
    };

private:
    Puzzle& Puzzle;
    PuzzleOptions& Opts;
    const int Segments;
    const uint64_t SegmentSize;
    const uint64_t SegmentMask;
    const int OperatorsCount;
    const int OperatorsMask;

    Store& CurFrontierStore;
    Store& NextFrontierStore;
    Store& CrossSegmentStore;
    SegmentReader FrontierReader;
    SegmentWriter FrontierWriter;
    SegmentReader CrossSegmentReader;
    Multiplexor Mult;
    ExpandBuffer Expander;

    MultiBitArray Array;
    BoolArray FrontierArray;
};

std::vector<uint64_t> DiskBasedFrontierSearch(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    Timer timer;
    const int OPS = puzzle.OperatorsCount();
    const int OPS_MASK = (1 << OPS) - 1;
    const uint64_t SIZE = puzzle.IndexesCount();
    uint64_t SEGMENT_SIZE = 1ui64 << opts.segmentBits;
    const uint64_t SEGMENT_MASK = SEGMENT_SIZE - 1;
    const int SEGMENTS = int((SIZE + SEGMENT_SIZE - 1) / SEGMENT_SIZE);
    if (SEGMENTS == 1 && SEGMENT_SIZE > SIZE) {
        SEGMENT_SIZE = SIZE; // SEGMENT_MASK is still valid
    }

    // classic FrontierSearch stores indexes and used operator bits in a single 4-byte word
    ensure(opts.segmentBits + OPS <= 32);

    std::cerr
        << "DiskBasedFrontierSearch"
        << "; nodes: " << WithDecSep(SIZE)
        << "; segments: " << WithDecSep(SEGMENTS)
        << "; segment size: " << WithDecSep(SEGMENT_SIZE) << std::endl;

    std::vector<uint64_t> result;

    Store curFrontierStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "frontier1");
    Store newFrontierStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "frontier2");
    Store crossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "xseg");

    std::vector<std::unique_ptr<DB_FS_Solver>> solvers;
    for (int i = 0; i < opts.threads; i++) {
        solvers.emplace_back(std::make_unique<DB_FS_Solver>(
            puzzle,
            opts,
            SEGMENTS,
            SEGMENT_SIZE,
            SEGMENT_MASK,
            OPS,
            OPS_MASK,
            curFrontierStore,
            newFrontierStore,
            crossSegmentStore
        ));
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

        {
            std::atomic<int> currentSegment{ 0 };

            auto fnExpand = [&](int index) {
                auto& solver = *(solvers[index]);
                while (true) {
                    int segment = currentSegment.fetch_add(1);
                    if (segment >= SEGMENTS) break;
                    solver.Expand(segment);
                }
            };

            std::vector<std::thread> threads;
            for (int i = 0; i < opts.threads; i++) {
                threads.emplace_back(fnExpand, i);
            }
            for (auto& thread : threads) thread.join();
        }

        {
            auto fnFinishExpand = [&](int index) {
                auto& solver = *(solvers[index]);
                solver.FinishExpand();
            };

            std::vector<std::thread> threads;
            for (int i = 0; i < opts.threads; i++) {
                threads.emplace_back(fnFinishExpand, i);
            }
            for (auto& thread : threads) thread.join();
        }

        uint64_t xstore_size = crossSegmentStore.TotalLength();

        {
            std::atomic<int> currentSegment{ 0 };

            auto fnCollect = [&](int index) {
                auto& solver = *(solvers[index]);
                while (true) {
                    int segment = currentSegment.fetch_add(1);
                    if (segment >= SEGMENTS) break;
                    totalCount += solver.Collect(segment);
                }
            };

            std::vector<std::thread> threads;
            for (int i = 0; i < opts.threads; i++) {
                threads.emplace_back(fnCollect, i);
            }
            for (auto& thread : threads) thread.join();
        }

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
            << ", x-seg (before)=" << WithSize(xstore_size)
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
