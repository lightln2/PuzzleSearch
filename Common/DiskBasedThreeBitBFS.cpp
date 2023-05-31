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
        store.Rewind(segment);
        auto read = store.ReadArray(segment, arr.Data(), arr.DataSize());
        ensure(read == 0 || read == arr.DataSize());
    };

    void SaveBoolArray(int segment, Store& store, BoolArray& arr) {
        store.WriteArray(segment, arr.Data(), arr.DataSize());
        arr.Clear();
    };


} // namespace

std::vector<uint64_t> DiskBasedThreeBitBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    std::cerr << "DB_3BFS" << std::endl;
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

    Store oldStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "store1");
    Store curStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "store2");
    Store newStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "store3");
    Store currentCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "xseg1");
    Store nextCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "xseg2");

    SegmentReader currentXSegReader(currentCrossSegmentStore);
    Multiplexor mult(nextCrossSegmentStore, SEGMENTS);

    BoolArray oldList(SEGMENT_SIZE);
    BoolArray curList(SEGMENT_SIZE);
    BoolArray newList(SEGMENT_SIZE);

    auto fnLoadOld = [&](int segment) {
        LoadBoolArray(segment, oldStore, oldList);
        oldStore.Delete(segment);
    };
    auto fnLoadCur = [&](int segment) {
        LoadBoolArray(segment, curStore, curList);
    };
    auto fnSave = [&](int segment) {
        SaveBoolArray(segment, newStore, newList);
    };

    auto fnGetSegIdx = [&](uint64_t index) {
        return std::pair<int, uint32_t>(index >> opts.segmentBits, index & SEGMENT_MASK);
    };

    ExpandBuffer nodes(puzzle);

    auto initialIndex = puzzle.Parse(initialState);
    auto [seg, idx] = fnGetSegIdx(initialIndex);
    newList.Set(idx);
    fnSave(seg);
    // expand cross-segment
    {
        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [s, idx] = fnGetSegIdx(child);
            if (s == seg) return;
            mult.Add(s, idx);
        };
        nodes.Add(initialIndex, 0, fnExpandCrossSegment);
        nodes.Finish(fnExpandCrossSegment);
        mult.FlushAllSegments();
    }
    std::swap(oldStore, curStore);
    std::swap(curStore, newStore);
    std::swap(currentCrossSegmentStore, nextCrossSegmentStore);

    result.push_back(1);

    uint64_t total_sz_old = 0;
    uint64_t total_sz_cur = 0;
    uint64_t total_sz_xseg = 0;

    std::cerr << "Step: 1; Count: 1" << std::endl;

    while (true) {
        Timer stepTimer;
        uint64_t totalCount = 0;

        for (int segment = 0; segment < SEGMENTS; segment++) {
            uint64_t indexBase = (uint64_t)segment << opts.segmentBits;

            fnLoadOld(segment);
            fnLoadCur(segment);
            currentXSegReader.SetSegment(segment);

            while (true) {
                auto& vect = currentXSegReader.Read();
                if (vect.IsEmpty()) break;
                for (size_t i = 0; i < vect.Size(); i++) {
                    uint32_t idx = vect[i];
                    newList.Set(idx);
                }
            }
            currentCrossSegmentStore.Delete(segment);

            auto fnExpandInSegment = [&](uint64_t child, int op) {
                auto [seg, idx] = fnGetSegIdx(child);
                if (seg != segment) return;
                newList.Set(idx);
            };

            curList.ScanBits([&](uint64_t index) {
                nodes.Add(indexBase | index, 0, fnExpandInSegment);
            });
            nodes.Finish(fnExpandInSegment);

            newList.AndNot(curList);
            newList.AndNot(oldList);

            auto fnExpandCrossSegment = [&](uint64_t child, int op) {
                auto [seg, idx] = fnGetSegIdx(child);
                if (seg == segment) return;
                mult.Add(seg, idx);
            };
            newList.ScanBits([&](uint64_t index) {
                nodes.Add(indexBase | index, 0, fnExpandCrossSegment);
            });
            nodes.Finish(fnExpandCrossSegment);

            mult.FlushAllSegments();

            totalCount += newList.BitsCount();
            fnSave(segment);
        }

        if (totalCount == 0) break;
        result.push_back(totalCount);
        oldStore.DeleteAll();
        currentCrossSegmentStore.DeleteAll();
        std::swap(oldStore, curStore);
        std::swap(curStore, newStore);
        std::swap(currentCrossSegmentStore, nextCrossSegmentStore);
        std::cerr
            << "Step: " << result.size()
            << "; count: " << totalCount
            << " in " << stepTimer
            << "; size: old=" << WithSize(oldStore.TotalLength())
            << ", cur=" << WithSize(curStore.TotalLength())
            << ", x-seg=" << WithSize(currentCrossSegmentStore.TotalLength())
            << std::endl;
        total_sz_old += oldStore.TotalLength();
        total_sz_cur += curStore.TotalLength();
        total_sz_xseg += currentCrossSegmentStore.TotalLength();
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
