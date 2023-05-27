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
        auto read = store.ReadArray(segment, &arr.Data()[0], arr.Data().size());
        ensure(read == 0 || read == arr.Data().size());
    };

    void SaveBoolArray(int segment, Store& store, BoolArray& arr) {
        store.WriteArray(segment, &arr.Data()[0], arr.Data().size());
        arr.Clear();
    };


} // namespace

std::vector<uint64_t> DiskBasedThreeBitBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    std::cerr << "DB_BFS" << std::endl;
    Timer timer;
    const uint64_t SIZE = puzzle.IndexesCount();
    uint64_t SEGMENT_SIZE = 1ui64 << opts.segmentBits;
    const uint64_t SEGMENT_MASK = SEGMENT_SIZE - 1;
    const int SEGMENTS = (SIZE + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
    if (SEGMENTS == 1 && SEGMENT_SIZE > SIZE) {
        SEGMENT_SIZE = SIZE; // SEGMENT_MASK is still valid
    }

    std::cerr
        << "total: " << WithDecSep(SIZE)
        << "; segments: " << WithDecSep(SEGMENTS)
        << "; segment size: " << WithDecSep(SEGMENT_SIZE) << std::endl;

    std::vector<uint64_t> result;

    std::vector<std::string> oldListDirs;
    std::vector<std::string> curListDirs;
    std::vector<std::string> newListDirs;
    std::vector<std::string> crossSegmentDirs1;
    std::vector<std::string> crossSegmentDirs2;
    for (const auto& dir : opts.directories) {
        oldListDirs.push_back(dir + "/old/");
        curListDirs.push_back(dir + "/cur/");
        newListDirs.push_back(dir + "/new/");
        crossSegmentDirs1.push_back(dir + "/xseg1/");
        crossSegmentDirs2.push_back(dir + "/xseg2/");
    }

    Store oldStore = Store::CreateMultiFileStore(SEGMENTS, oldListDirs);
    Store curStore = Store::CreateMultiFileStore(SEGMENTS, curListDirs);
    Store newStore = Store::CreateMultiFileStore(SEGMENTS, newListDirs);
    Store currentCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, crossSegmentDirs1);
    Store nextCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, crossSegmentDirs2);

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
    //TODO: expand cross-segment!!!
    //std::swap(oldStore, curStore);
    std::swap(curStore, newStore);
    std::swap(currentCrossSegmentStore, nextCrossSegmentStore);
    result.push_back(1);

    uint64_t total_sz_cur = 0;
    uint64_t total_sz_new = 0;
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
                if (vect.empty()) break;
                for (uint32_t idx : vect) {
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

            newList.AndNot(curList);
            newList.AndNot(oldList);
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
            << "; size: cur=" << WithSize(curStore.TotalLength())
            << ", new=" << WithSize(newStore.TotalLength())
            << ", x-seg=" << WithSize(currentCrossSegmentStore.TotalLength())
            << std::endl;
        total_sz_cur += curStore.TotalLength();
        total_sz_new += newStore.TotalLength();
        total_sz_xseg += currentCrossSegmentStore.TotalLength();
    }

    std::cerr << "Time: " << timer << std::endl;
    Store::PrintStats();
    std::cerr
        << "Total files: cur=" << WithSize(total_sz_cur)
        << "; new=" << WithSize(total_sz_new)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    return result;
}
