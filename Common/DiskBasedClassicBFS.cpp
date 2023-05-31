#include "BoolArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
#include "Util.h"

namespace {

    void LoadBoolArray(int segment, Store& store, BoolArray& arr) {
        arr.Clear();
        auto read = store.ReadArray(segment, arr.Data(), arr.DataSize());
        ensure(read == 0 || read == arr.DataSize());
        store.Delete(segment);
    };

    void SaveBoolArray(int segment, Store& store, BoolArray& arr) {
        store.WriteArray(segment, arr.Data(), arr.DataSize());
        arr.Clear();
    };


} // namespace

class DB_BFS_Solver {
public:
    DB_BFS_Solver(
        Puzzle& puzzle,
        PuzzleOptions& opts,
        int segments,
        uint64_t segmentSize,
        uint64_t segmentMask,
        Store& currentOpenListStore,
        Store& nextOpenListStore,
        Store& currentClosedListStore,
        Store& nextClosedListStore,
        Store& currentCrossSegmentStore,
        Store& nextCrossSegmentStore)
        
        : Puzzle(puzzle)
        , Opts(opts)
        , Segments(segments)
        , SegmentSize(segmentSize)
        , SegmentMask(segmentMask)
        , CurrentOpenListStore(currentOpenListStore)
        , NextOpenListStore(nextOpenListStore)
        , CurrentClosedListStore(currentClosedListStore)
        , NextClosedListStore(nextClosedListStore)
        , CurrentCrossSegmentStore(currentCrossSegmentStore)
        , NextCrossSegmentStore(nextCrossSegmentStore)
        , CurrentXSegReader(currentCrossSegmentStore)
        , Mult(nextCrossSegmentStore, Segments)
        , Expander(puzzle)
        , ClosedList(SegmentSize)
        , OpenList(SegmentSize)
        , NewOpenList(SegmentSize)
    {
        std::cerr << "DiskBasedClassicBFS"
            << "; nodes: " << WithDecSep(Puzzle.IndexesCount())
            << "; segments: " << WithDecSep(Segments)
            << "; segment size: " << WithDecSep(SegmentSize)
            << std::endl;
    }

    void SetInitialNode(const std::string& initialState) {
        auto initialIndex = Puzzle.Parse(initialState);
        auto [seg, idx] = GetSegIdx(initialIndex);
        NewOpenList.Set(idx);
        SaveOpenList(seg);
        std::swap(CurrentOpenListStore, NextOpenListStore);
    }

    uint64_t Process(int segment) {
        uint64_t indexBase = (uint64_t)segment << Opts.segmentBits;
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
        CurrentCrossSegmentStore.Delete(segment);

        uint64_t totalCount = OpenList.AndNotAndCount(ClosedList);
        ClosedList.Or(OpenList);

        auto fnExpand = [&](uint64_t child, int op) {
            auto [seg, idx] = GetSegIdx(child);
            if (seg == segment) NewOpenList.Set(idx);
            else Mult.Add(seg, idx);
        };

        OpenList.ScanBitsAndClear([&](uint64_t index) {
            Expander.Add(indexBase | index, 0, fnExpand);
        });
        Expander.Finish(fnExpand);

        Mult.FlushAllSegments();

        NewOpenList.AndNot(ClosedList);

        SaveOpenList(segment);
        SaveClosedList(segment);

        return totalCount;
    }

private:
    void LoadClosedList(int segment) {
        LoadBoolArray(segment, CurrentClosedListStore, ClosedList);
    }

    void SaveClosedList(int segment) {
        SaveBoolArray(segment, NextClosedListStore, ClosedList);
    };

    void LoadOpenList(int segment) {
        LoadBoolArray(segment, CurrentOpenListStore, OpenList);
    };

    void SaveOpenList(int segment) {
        SaveBoolArray(segment, NextOpenListStore, NewOpenList);
    };

    std::pair<int, uint32_t> GetSegIdx(uint64_t index) {
        return { int(index >> Opts.segmentBits), uint32_t(index & SegmentMask) };
    };

private:
    Puzzle& Puzzle;
    PuzzleOptions& Opts;
    const int Segments;
    const uint64_t SegmentSize;
    const uint64_t SegmentMask;

    Store& CurrentOpenListStore;
    Store& NextOpenListStore;
    Store& CurrentClosedListStore;
    Store& NextClosedListStore;
    Store& CurrentCrossSegmentStore;
    Store& NextCrossSegmentStore;
    SegmentReader CurrentXSegReader;
    Multiplexor Mult;
    ExpandBuffer Expander;

    BoolArray ClosedList;
    BoolArray OpenList;
    BoolArray NewOpenList;
};


std::vector<uint64_t> DiskBasedClassicBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    ensure(opts.segmentBits <= 32);
    Timer timer;
    const uint64_t SIZE = puzzle.IndexesCount();
    uint64_t SEGMENT_SIZE = 1ui64 << opts.segmentBits;
    uint64_t SEGMENT_MASK = SEGMENT_SIZE - 1;
    const int SEGMENTS = int((SIZE + SEGMENT_SIZE - 1) / SEGMENT_SIZE);
    if (SEGMENTS == 1 && SEGMENT_SIZE > SIZE) {
        SEGMENT_SIZE = SIZE; // SEGMENT_MASK is still valid
    }

    std::vector<uint64_t> result;

    Store currentOpenListStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "open1");
    Store nextOpenListStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "open2");
    Store currentClosedListStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "closed1");
    Store nextClosedListStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "closed2");
    Store currentCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "xseg1");
    Store nextCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "xseg2");

    DB_BFS_Solver solver(
        puzzle,
        opts,
        SEGMENTS,
        SEGMENT_SIZE,
        SEGMENT_MASK,
        currentOpenListStore,
        nextOpenListStore,
        currentClosedListStore,
        nextClosedListStore,
        currentCrossSegmentStore,
        nextCrossSegmentStore);


    solver.SetInitialNode(initialState);

    uint64_t total_sz_open = 0;
    uint64_t total_sz_closed = 0;
    uint64_t total_sz_xseg = 0;

    while (true) {
        Timer stepTimer;
        uint64_t totalCount = 0;

        for (int segment = 0; segment < SEGMENTS; segment++) {
            totalCount += solver.Process(segment);
        }

        if (totalCount == 0) break;
        result.push_back(totalCount);
        currentOpenListStore.DeleteAll();
        currentClosedListStore.DeleteAll();
        currentCrossSegmentStore.DeleteAll();
        std::swap(currentOpenListStore, nextOpenListStore);
        std::swap(currentClosedListStore, nextClosedListStore);
        std::swap(currentCrossSegmentStore, nextCrossSegmentStore);
        std::cerr 
            << "Step: " << result.size() 
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
