#include "BitArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
#include "ThreadUtil.h"
#include "Util.h"

namespace {

    struct TwoBitArray {
    public:
        TwoBitArray(uint64_t size) : array(size * 2) {}

        int Get(uint64_t index) {
            uint64_t offset = index / 32;
            uint64_t pos = 2 * (index % 32);
            return (array.Data()[offset] >> pos) & 3;
        }

        void Set(uint64_t index, int value) {
            uint64_t offset = index / 32;
            uint64_t pos = 2 * (index % 32);
            array.Data()[offset] &= ~(3ui64 << pos);
            array.Data()[offset] |= ((uint64_t)value << pos);
        }

        void Load(int segment, Store& store) {
            array.Clear();
            auto read = store.ReadArray(segment, array.Data(), array.DataSize());
            ensure(read == 0 || read == array.DataSize());
            store.Delete(segment);
        }

        void Save(int segment, Store& store) {
            store.WriteArray(segment, &array.Data()[0], array.DataSize());
            array.Clear();
        }

    private:
        BitArray array;
    };


} // namespace

class DB_2BitBFS_Solver {
public:
    static int UNVISITED, OLD, CUR, NEXT;
public:
    DB_2BitBFS_Solver(
        SegmentedOptions sopts,
        Store& curArrStore,
        Store& nextArrStore,
        Store& curCrossSegmentStore,
        Store& nextCrossSegmentStore)

        : SOpts(sopts)
        , CurArrStore(curArrStore)
        , NextArrStore(nextArrStore)
        , CurCrossSegmentStore(curCrossSegmentStore)
        , NextCrossSegmentStore(nextCrossSegmentStore)
        , CrossSegmentReader(curCrossSegmentStore)
        , Mult(nextCrossSegmentStore, SOpts.Segments)
        , Expander(SOpts.Puzzle)
        , Array(SOpts.SegmentSize)
    {
    }

    void SetInitialNode(const std::string& initialState) {
        auto initialIndex = SOpts.Puzzle.Parse(initialState);
        auto [seg, idx] = SOpts.GetSegIdx(initialIndex);
        Array.Set(idx, CUR);
        Save(seg);
        std::swap(CurArrStore, NextArrStore);
    }

    uint64_t Process(int segment) {
        uint64_t indexBase = (uint64_t)segment << SOpts.Opts.segmentBits;

        Load(segment);
        CrossSegmentReader.SetSegment(segment);

        while (true) {
            auto& vect = CrossSegmentReader.Read();
            if (vect.IsEmpty()) break;
            for (size_t i = 0; i < vect.Size(); i++) {
                uint32_t idx = vect[i];
                int val = Array.Get(idx);
                if (val == OLD) continue;
                Array.Set(idx, CUR);
            }
        }
        CurCrossSegmentStore.Delete(segment);

        auto fnExpand = [&](uint64_t child, int op) {
            auto [seg, idx] = SOpts.GetSegIdx(child);
            if (seg == segment) {
                if (Array.Get(idx) == UNVISITED) {
                    Array.Set(idx, NEXT);
                }
            }
            else Mult.Add(seg, idx);
        };

        uint64_t count = 0;

        for (uint64_t i = 0; i < SOpts.SegmentSize; i++) {
            int val = Array.Get(i);
            if (val != CUR) continue;
            count++;
            Array.Set(i, OLD);
            Expander.Add(indexBase | i, 0, fnExpand);
        }
        Expander.Finish(fnExpand);

        Save(segment);

        return count;
    }

    void FinishProcess() {
        Mult.FlushAllSegments();
    }

private:
    void Load(int segment) {
        Array.Load(segment, CurArrStore);
    };

    void Save(int segment) {
        Array.Save(segment, NextArrStore);
    };

private:
    SegmentedOptions SOpts;
    Store& CurArrStore;
    Store& NextArrStore;
    Store& CurCrossSegmentStore;
    Store& NextCrossSegmentStore;
    SegmentReader CrossSegmentReader;
    Multiplexor Mult;
    ExpandBuffer Expander;

    TwoBitArray Array;
};

int DB_2BitBFS_Solver::UNVISITED = 0;
int DB_2BitBFS_Solver::OLD = 1;
int DB_2BitBFS_Solver::CUR = 2;
int DB_2BitBFS_Solver::NEXT = 3;


std::vector<uint64_t> DiskBasedTwoBitBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    std::cerr << "DiskBaset TwoBit BFS" << std::endl;
    Timer timer;
    ensure(opts.segmentBits <= 32);
    SegmentedOptions sopts(puzzle, opts);
    sopts.PrintOptions();

    std::vector<uint64_t> result;

    Store currentArrStore = sopts.MakeStore("arr1");
    Store nextArrStore = sopts.MakeStore("arr2");
    Store currentCrossSegmentStore = sopts.MakeStore("xseg1");
    Store nextCrossSegmentStore = sopts.MakeStore("xseg2");

    std::vector<std::unique_ptr<DB_2BitBFS_Solver>> solvers;
    for (int i = 0; i < opts.threads; i++) {
        solvers.emplace_back(std::make_unique<DB_2BitBFS_Solver>(
            sopts,
            currentArrStore,
            nextArrStore,
            currentCrossSegmentStore,
            nextCrossSegmentStore));
    }

    solvers[0]->SetInitialNode(initialState);

    uint64_t total_sz_arr = 0;
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
        currentArrStore.DeleteAll();
        currentCrossSegmentStore.DeleteAll();
        std::swap(DB_2BitBFS_Solver::CUR, DB_2BitBFS_Solver::NEXT);
        std::swap(currentArrStore, nextArrStore);
        std::swap(currentCrossSegmentStore, nextCrossSegmentStore);
        std::cerr
            << "Step: " << result.size()
            << "; count: " << totalCount
            << " in " << stepTimer
            << "; size: arr=" << WithSize(currentArrStore.TotalLength())
            << ", x-seg=" << WithSize(currentCrossSegmentStore.TotalLength())
            << std::endl;
        total_sz_arr += currentArrStore.TotalLength();
        total_sz_xseg += currentCrossSegmentStore.TotalLength();
    }

    std::cerr << "Time: " << timer << std::endl;
    Store::PrintStats();
    std::cerr
        << "Total files: arr=" << WithSize(total_sz_arr)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    return result;
}
