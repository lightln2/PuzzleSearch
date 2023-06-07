#include "BitArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
#include "ThreadUtil.h"
#include "Util.h"

template<int BITS>
class DB_FS2_Solver {
public:
    DB_FS2_Solver(
        SegmentedOptions& sopts,
        Store& curFrontierStore,
        Store& nextFrontierStore,
        Store& curOpBitsStore,
        Store& nextOpBitsStore,
        std::vector<Store>& curCrossSegmentStores,
        std::vector<Store>& nextCrossSegmentStores)

        : SOpts(sopts)
        , CurFrontierStore(curFrontierStore)
        , NextFrontierStore(nextFrontierStore)
        , CurOpBitsStore(curOpBitsStore)
        , NextOpBitsStore(nextOpBitsStore)
        , CurCrossSegmentStores(curCrossSegmentStores)
        , NextCrossSegmentStores(nextCrossSegmentStores)
        , FrontierReader(curFrontierStore)
        , FrontierWriter(nextFrontierStore)
        , OpBitsReader(curOpBitsStore)
        , OpBitsWriter(nextOpBitsStore)
        , Array(SOpts.SegmentSize)
        , FrontierArray(SOpts.SegmentSize)
        , Expander(SOpts.Puzzle)
    {
        for (auto& store : CurCrossSegmentStores) {
            CrossSegmentReaders.emplace_back(std::make_unique<SegmentReader>(store));
        }
        for (auto& store : NextCrossSegmentStores) {
            Mults.emplace_back(std::make_unique<Multiplexor>(store, SOpts.Segments));
        }
    }

    void SetInitialNode(const std::string& initialState) {
        auto initialIndex = SOpts.Puzzle.Parse(initialState);
        auto [seg, idx] = SOpts.GetSegIdx(initialIndex);
        FrontierWriter.SetSegment(seg);
        OpBitsWriter.SetSegment(seg);
        FrontierWriter.Add(idx);
        OpBitsWriter.Add(0);
        FrontierWriter.Flush();
        OpBitsWriter.Flush();

        //Expand cross-segment
        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [s, idx] = SOpts.GetSegIdx(child);
            if (s == seg) return;
            Mults[op]->Add(s, idx);
        };

        Expander.Add(initialIndex, 0, fnExpandCrossSegment);
        Expander.Finish(fnExpandCrossSegment);
        for (int i = 0; i < SOpts.OperatorsCount; i++)
            Mults[i]->FlushAllSegments();
        std::swap(CurFrontierStore, NextFrontierStore);
        std::swap(CurOpBitsStore, NextOpBitsStore);
        for (int i = 0; i < SOpts.OperatorsCount; i++)
            std::swap(CurCrossSegmentStores[i], NextCrossSegmentStores[i]);
    }

    uint64_t Expand(int segment) {
        uint64_t indexBase = (uint64_t)segment << SOpts.Opts.segmentBits;

        bool hasData = false;

        for (int i = 0; i < SOpts.OperatorsCount; i++)
            CrossSegmentReaders[i]->SetSegment(segment);
        FrontierReader.SetSegment(segment);
        FrontierWriter.SetSegment(segment);
        OpBitsReader.SetSegment(segment);
        OpBitsWriter.SetSegment(segment);

        for (int op = 0; op < SOpts.OperatorsCount; op++) {
            while (true) {
                auto& vect = CrossSegmentReaders[op]->Read();
                if (vect.IsEmpty()) break;
                hasData = true;
                for (size_t i = 0; i < vect.Size(); i++) {
                    uint32_t idx = vect[i];
                    Array.Set(idx, op);
                }
            }
            CurCrossSegmentStores[op].Delete(segment);
        }

        auto fnExpandInSegment = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg != segment) return;
            Array.Set(idx, op);
        };

        while (true) {
            auto& vect = FrontierReader.Read();
            if (vect.IsEmpty()) break;
            auto& opVect = OpBitsReader.Read();
            ensure(vect.Size() == opVect.Size());
            hasData = true;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect[i];
                uint8_t opBits = opVect[i];
                Expander.Add(indexBase | idx, opBits, fnExpandInSegment);
                if (SOpts.Puzzle.HasOddLengthCycles()) FrontierArray.Set(idx);
            }
        }

        if (!hasData) return 0;

        Expander.Finish(fnExpandInSegment);
        CurFrontierStore.Delete(segment);
        CurOpBitsStore.Delete(segment);

        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg == segment) return;
            Mults[op]->Add(seg, idx);
        };

        uint64_t count = 0;

        Array.ScanBitsAndClear([&](uint64_t index, int opBits) {
            if (SOpts.HasOddLengthCycles && FrontierArray.Get(index)) return;
            count++;
            if (opBits < SOpts.OperatorsMask) {
                Expander.Add(indexBase | index, opBits, fnExpandCrossSegment);
                FrontierWriter.Add(uint32_t(index));
                OpBitsWriter.Add(uint8_t(opBits));
            }
        });
        Expander.Finish(fnExpandCrossSegment);
        FrontierWriter.Flush();
        OpBitsWriter.Flush();

        if (SOpts.HasOddLengthCycles) FrontierArray.Clear();

        return count;
    }

    void FinishExpand() {
        for (int op = 0; op < SOpts.OperatorsCount; op++)
            Mults[op]->FlushAllSegments();
    }

private:
    SegmentedOptions SOpts;
    Store& CurFrontierStore;
    Store& NextFrontierStore;
    Store& CurOpBitsStore;
    Store& NextOpBitsStore;
    std::vector<Store>& CurCrossSegmentStores;
    std::vector<Store>& NextCrossSegmentStores;
    SegmentReader FrontierReader;
    SegmentWriter FrontierWriter;
    OpBitsReader OpBitsReader;
    OpBitsWriter OpBitsWriter;
    std::vector<std::unique_ptr<SegmentReader>> CrossSegmentReaders;
    std::vector<std::unique_ptr<Multiplexor>> Mults;
    ExpandBuffer Expander;

    MultiBitArray<BITS> Array;
    BitArray FrontierArray;
};

template<int BITS>
std::vector<uint64_t> DiskBasedFrontierSearch2Int(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    Timer timer;
    std::cerr << "DiskBasedFrontierSearch2" << std::endl;

    SegmentedOptions sopts(puzzle, opts);

    ensure(opts.segmentBits <= 32);
    ensure(sopts.OperatorsCount <= 8);

    sopts.PrintOptions();

    std::vector<uint64_t> result;

    Store curFrontierStore = sopts.MakeStore("frontier1");
    Store newFrontierStore = sopts.MakeStore("frontier2");
    Store curOpBitsStore = sopts.MakeStore("opBits1");
    Store newOpBitsStore = sopts.MakeStore("opBits2");
    std::vector<Store> curCrossSegmentStores;
    std::vector<Store> nextCrossSegmentStores;
    for (int op = 0; op < sopts.OperatorsCount; op++) {
        curCrossSegmentStores.emplace_back(sopts.MakeStore("xseg1_" + std::to_string(op) + "_"));
        nextCrossSegmentStores.emplace_back(sopts.MakeStore("xseg2_" + std::to_string(op) + "_"));
    }

    std::vector<std::unique_ptr<DB_FS2_Solver<BITS>>> solvers;
    for (int i = 0; i < opts.threads; i++) {
        solvers.emplace_back(std::make_unique<DB_FS2_Solver<BITS>>(
            sopts,
            curFrontierStore,
            newFrontierStore,
            curOpBitsStore,
            newOpBitsStore,
            curCrossSegmentStores,
            nextCrossSegmentStores));
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
        curOpBitsStore.DeleteAll();
        std::swap(curFrontierStore, newFrontierStore);
        std::swap(curOpBitsStore, newOpBitsStore);
        uint64_t x_total_len = 0;
        for (int op = 0; op < sopts.OperatorsCount; op++) {
            curCrossSegmentStores[op].DeleteAll();
            std::swap(curCrossSegmentStores[op], nextCrossSegmentStores[op]);
            x_total_len += curCrossSegmentStores[op].TotalLength();
        }
        std::cerr
            << "Step: " << result.size() - 1
            << "; count: " << WithDecSep(totalCount)
            << " in " << stepTimer
            << "; size: frontier=" << WithSize(curFrontierStore.TotalLength() + curOpBitsStore.TotalLength())
            << ", x-seg=" << WithSize(x_total_len)
            << std::endl;
        total_sz_frontier += curFrontierStore.TotalLength() + curOpBitsStore.TotalLength();
        total_sz_xseg += x_total_len;
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

std::vector<uint64_t> DiskBasedFrontierSearch2(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    int bits = puzzle.OperatorsCount();
    ensure(bits > 0 && bits <= 16);
    if (bits == 1)
        return DiskBasedFrontierSearch2Int<1>(puzzle, initialState, opts);
    else if (bits == 2)
        return DiskBasedFrontierSearch2Int<2>(puzzle, initialState, opts);
    else if (bits <= 4)
        return DiskBasedFrontierSearch2Int<4>(puzzle, initialState, opts);
    else if (bits <= 8)
        return DiskBasedFrontierSearch2Int<8>(puzzle, initialState, opts);
    else
        return DiskBasedFrontierSearch2Int<16>(puzzle, initialState, opts);
}
