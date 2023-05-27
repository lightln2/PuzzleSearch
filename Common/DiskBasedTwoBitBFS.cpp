#include "BoolArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
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
            auto read = store.ReadArray(segment, &array.Data()[0], array.Data().size());
            ensure(read == 0 || read == array.Data().size());
            store.Delete(segment);
        }

        void Save(int segment, Store& store) {
            store.WriteArray(segment, &array.Data()[0], array.Data().size());
            array.Clear();
        }

    private:
        BoolArray array;
    };


} // namespace

std::vector<uint64_t> DiskBasedTwoBitBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    std::cerr << "2BIT_BFS" << std::endl;
    Timer timer;
    const uint64_t SIZE = puzzle.IndexesCount();
    uint64_t SEGMENT_SIZE = 1ui64 << opts.segmentBits;
    const uint64_t SEGMENT_MASK = SEGMENT_SIZE - 1;
    const int SEGMENTS = (SIZE + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
    if (SEGMENTS == 1 && SEGMENT_SIZE > SIZE) {
        SEGMENT_SIZE = SIZE; // SEGMENT_MASK is still valid
    }

    int UNVISITED = 0, OLD = 1, CUR = 2, NEXT = 3;

    std::cerr
        << "total: " << WithDecSep(SIZE)
        << "; segments: " << WithDecSep(SEGMENTS)
        << "; segment size: " << WithDecSep(SEGMENT_SIZE) << std::endl;

    std::vector<uint64_t> result;

    std::vector<std::string> arrayDirs1;
    std::vector<std::string> arrayDirs2;
    std::vector<std::string> crossSegmentDirs1;
    std::vector<std::string> crossSegmentDirs2;
    for (const auto& dir : opts.directories) {
        arrayDirs1.push_back(dir + "/arr1/");
        arrayDirs2.push_back(dir + "/arr2/");
        crossSegmentDirs1.push_back(dir + "/xseg1/");
        crossSegmentDirs2.push_back(dir + "/xseg2/");
    }

    Store currentArrStore = Store::CreateMultiFileStore(SEGMENTS, arrayDirs1);
    Store nextArrStore = Store::CreateMultiFileStore(SEGMENTS, arrayDirs2);
    Store currentCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, crossSegmentDirs1);
    Store nextCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, crossSegmentDirs2);

    TwoBitArray array(SEGMENT_SIZE);

    SegmentReader currentXSegReader(currentCrossSegmentStore);
    Multiplexor mult(nextCrossSegmentStore, SEGMENTS);

    auto fnLoad = [&](int segment) {
        array.Load(segment, currentArrStore);
    };

    auto fnSave = [&](int segment) {
        array.Save(segment, nextArrStore);
    };

    auto fnGetSegIdx = [&](uint64_t index) {
        return std::pair<int, uint32_t>(index >> opts.segmentBits, index & SEGMENT_MASK);
    };

    auto initialIndex = puzzle.Parse(initialState);
    auto [seg, idx] = fnGetSegIdx(initialIndex);
    array.Set(idx, CUR);
    fnSave(seg);
    std::swap(currentArrStore, nextArrStore);

    ExpandBuffer nodes(puzzle);

    uint64_t total_sz_arr = 0;
    uint64_t total_sz_xseg = 0;

    while (true) {
        Timer stepTimer;
        uint64_t totalCount = 0;

        for (int segment = 0; segment < SEGMENTS; segment++) {
            uint64_t indexBase = (uint64_t)segment << opts.segmentBits;

            fnLoad(segment);
            currentXSegReader.SetSegment(segment);

            while (true) {
                auto& vect = currentXSegReader.Read();
                if (vect.empty()) break;
                for (uint32_t idx : vect) {
                    int val = array.Get(idx);
                    if (val == OLD) continue;
                    array.Set(idx, CUR);
                }
            }
            currentCrossSegmentStore.Delete(segment);

            auto fnExpand = [&](uint64_t child, int op) {
                auto [seg, idx] = fnGetSegIdx(child);
                if (seg == segment) {
                    if (array.Get(idx) == UNVISITED) {
                        array.Set(idx, NEXT);
                    }
                }
                else mult.Add(seg, idx);
            };

            uint64_t count = 0;

            for (uint64_t i = 0; i < SEGMENT_SIZE; i++) {
                int val = array.Get(i);
                if (val != CUR) continue;
                count++;
                array.Set(i, OLD);
                nodes.Add(indexBase | i, 0, fnExpand);
            }
            nodes.Finish(fnExpand);

            mult.FlushAllSegments();

            fnSave(segment);

            totalCount += count;

        }

        if (totalCount == 0) break;
        result.push_back(totalCount);
        std::swap(CUR, NEXT);
        currentArrStore.DeleteAll();
        currentCrossSegmentStore.DeleteAll();
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
