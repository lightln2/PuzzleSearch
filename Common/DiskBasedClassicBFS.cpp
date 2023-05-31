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

std::vector<uint64_t> DiskBasedClassicBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    std::cerr << "DB_BFS" << std::endl;
    ensure(opts.segmentBits <= 32);
    Timer timer;
    const uint64_t SIZE = puzzle.IndexesCount();
    uint64_t SEGMENT_SIZE = 1ui64 << opts.segmentBits;
    const uint64_t SEGMENT_MASK = SEGMENT_SIZE - 1;
    const int SEGMENTS = int((SIZE + SEGMENT_SIZE - 1) / SEGMENT_SIZE);
    if (SEGMENTS == 1 && SEGMENT_SIZE > SIZE) {
        SEGMENT_SIZE = SIZE; // SEGMENT_MASK is still valid
    }

    std::cerr
        << "total: " << WithDecSep(SIZE)
        << "; segments: " << WithDecSep(SEGMENTS)
        << "; segment size: " << WithDecSep(SEGMENT_SIZE) << std::endl;

    std::vector<uint64_t> result;

    Store currentOpenListStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "open1");
    Store nextOpenListStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "open2");
    Store currentClosedListStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "closed1");
    Store nextClosedListStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "closed2");
    Store currentCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "xseg1");
    Store nextCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "xseg2");

    SegmentReader currentXSegReader(currentCrossSegmentStore);
    Multiplexor mult(nextCrossSegmentStore, SEGMENTS);

    BoolArray closedList(SEGMENT_SIZE);
    BoolArray openList(SEGMENT_SIZE);
    BoolArray newOpenList(SEGMENT_SIZE);

    auto fnLoadClosedList = [&](int segment) {
        LoadBoolArray(segment, currentClosedListStore, closedList);
    };
    auto fnSaveClosedList = [&](int segment) {
        SaveBoolArray(segment, nextClosedListStore, closedList);
    };
    auto fnLoadOpenList = [&](int segment) {
        LoadBoolArray(segment, currentOpenListStore, openList);
    };
    auto fnSaveOpenList = [&](int segment) {
        SaveBoolArray(segment, nextOpenListStore, newOpenList);
    };

    auto fnGetSegIdx = [&](uint64_t index) {
        return std::pair<int, uint32_t>(int(index >> opts.segmentBits), uint32_t(index & SEGMENT_MASK));
    };

    auto initialIndex = puzzle.Parse(initialState);
    auto [seg, idx] = fnGetSegIdx(initialIndex);
    newOpenList.Set(idx);
    fnSaveOpenList(seg);
    std::swap(currentOpenListStore, nextOpenListStore);

    ExpandBuffer nodes(puzzle);

    uint64_t total_sz_open = 0;
    uint64_t total_sz_closed = 0;
    uint64_t total_sz_xseg = 0;

    while (true) {
        Timer stepTimer;
        uint64_t totalCount = 0;

        for (int segment = 0; segment < SEGMENTS; segment++) {
            uint64_t indexBase = (uint64_t)segment << opts.segmentBits;

            fnLoadOpenList(segment);
            fnLoadClosedList(segment);
            currentXSegReader.SetSegment(segment);

            while (true) {
                auto& vect = currentXSegReader.Read();
                if (vect.IsEmpty()) break;
                for (size_t i = 0; i < vect.Size(); i++) {
                    uint32_t idx = vect[i];
                    openList.Set(idx);
                }
            }
            currentCrossSegmentStore.Delete(segment);

            totalCount += openList.AndNotAndCount(closedList);
            closedList.Or(openList);

            auto fnExpand = [&](uint64_t child, int op) {
                auto [seg, idx] = fnGetSegIdx(child);
                if (seg == segment) newOpenList.Set(idx);
                else mult.Add(seg, idx);
            };

            openList.ScanBitsAndClear([&](uint64_t index) {
                nodes.Add(indexBase | index, 0, fnExpand);
            });
            nodes.Finish(fnExpand);

            mult.FlushAllSegments();

            newOpenList.AndNot(closedList);

            fnSaveOpenList(segment);
            fnSaveClosedList(segment);
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
    std::cerr
        << "Total files: open=" << WithSize(total_sz_open)
        << "; closed=" << WithSize(total_sz_closed)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    return result;
}
