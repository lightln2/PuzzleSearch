#include "BoolArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
#include "Util.h"

std::vector<uint64_t> DiskBasedClassicBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    std::cerr << "DB_BFS" << std::endl;
    Timer timer;
    const uint64_t SIZE = puzzle.IndexesCount();
    const uint64_t SEGMENT_SIZE = 1ui64 << opts.segmentBits;
    const uint64_t SEGMENT_MASK = SEGMENT_SIZE - 1;
    const int SEGMENTS = (SIZE + SEGMENT_SIZE - 1) / SEGMENT_SIZE;

    std::cerr << "segments: " << SEGMENTS << std::endl;

    std::vector<uint64_t> result;

    std::vector<std::string> closedListDirs1;
    std::vector<std::string> closedListDirs2;
    std::vector<std::string> openListDirs1;
    std::vector<std::string> openListDirs2;
    std::vector<std::string> crossSegmentDirs1;
    std::vector<std::string> crossSegmentDirs2;
    for (const auto& dir : opts.directories) {
        closedListDirs1.push_back(dir + "/closed/");
        closedListDirs2.push_back(dir + "/closed2/");
        openListDirs1.push_back(dir + "/open1/");
        openListDirs2.push_back(dir + "/open2/");
        crossSegmentDirs1.push_back(dir + "/xseg1/");
        crossSegmentDirs2.push_back(dir + "/xseg2/");
    }

    Store currentClosedListStore = Store::CreateMultiFileStore(SEGMENTS, closedListDirs1);
    Store nextClosedListStore = Store::CreateMultiFileStore(SEGMENTS, closedListDirs2);
    Store currentOpenListStore = Store::CreateMultiFileStore(SEGMENTS, openListDirs1);
    Store nextOpenListStore = Store::CreateMultiFileStore(SEGMENTS, openListDirs2);
    Store currentCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, crossSegmentDirs1);
    Store nextCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, crossSegmentDirs2);

    SegmentReader currentXSegReader(currentCrossSegmentStore);
    Multiplexor mult(nextCrossSegmentStore, SEGMENTS);

    BoolArray closedList(SEGMENT_SIZE);
    BoolArray openList(SEGMENT_SIZE);
    BoolArray newOpenList(SEGMENT_SIZE);

    auto fnLoadBoolArray = [](int segment, Store& store, BoolArray& arr) {
        arr.Clear();
        auto read = store.ReadArray(segment, &arr.Data()[0], arr.Data().size());
        ensure(read == 0 || read == arr.Data().size());
        store.Delete(segment);
    };

    auto fnSaveBoolArray = [](int segment, Store& store, BoolArray& arr) {
        store.WriteArray(segment, &arr.Data()[0], arr.Data().size());
        arr.Clear();
    };

    auto fnLoadClosedList = [&](int segment) {
        fnLoadBoolArray(segment, currentClosedListStore, closedList);
    };
    auto fnSaveClosedList = [&](int segment) {
        fnSaveBoolArray(segment, nextClosedListStore, closedList);
    };
    auto fnLoadOpenList = [&](int segment) {
        fnLoadBoolArray(segment, currentOpenListStore, openList);
    };
    auto fnSaveOpenList = [&](int segment) {
        fnSaveBoolArray(segment, nextOpenListStore, newOpenList);
    };

    auto fnGetSegIdx = [&](uint64_t index) {
        return std::pair<int, uint32_t>(index >> opts.segmentBits, index & SEGMENT_MASK);
    };

    auto initialIndex = puzzle.Parse(initialState);
    auto [seg, idx] = fnGetSegIdx(initialIndex);

    newOpenList.Set(idx);
    //closedList.Set(idx);
    fnSaveOpenList(seg);
    //fnSaveClosedList(seg);
    std::swap(currentOpenListStore, nextOpenListStore);
    //std::swap(currentClosedListStore, nextClosedListStore);
    //std::swap(currentCrossSegmentStore, nextCrossSegmentStore);

    //result.push_back(1);

    ExpandBuffer nodes(puzzle);

    //std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {

        uint64_t totalCount = 0;

        for (int segment = 0; segment < SEGMENTS; segment++) {
            uint64_t indexBase = (uint64_t)segment << opts.segmentBits;

            fnLoadOpenList(segment);
            fnLoadClosedList(segment);
            currentXSegReader.SetSegment(segment);
            //std::cerr << "ST segment: " << segment << "; openList: " << openList.BitsCount() << "; closedList: " << closedList.BitsCount() << std::endl;

            while (true) {
                auto& vect = currentXSegReader.Read();
                if (vect.empty()) break;
                for (uint32_t idx : vect) {
                    //std::cerr << "RD: " << segment << ":" << idx << std::endl;
                    openList.Set(idx);
                }
            }
            currentCrossSegmentStore.Delete(segment);

            totalCount += openList.AndNotAndCount(closedList);
            closedList.Or(openList);

            auto fnExpand = [&](uint64_t child, int op) {
                auto [seg, idx] = fnGetSegIdx(child);
                if (seg == segment) newOpenList.Set(idx);
                else {
                    //std::cerr << "XSEG: " << seg << ":" << idx << std::endl;
                    mult.Add(seg, idx);
                }
            };

            openList.ScanBitsAndClear([&](uint64_t index) {
                nodes.Add(indexBase | index, 0, fnExpand);
            });
            nodes.Finish(fnExpand);

            mult.FlushAllSegments();

            newOpenList.AndNot(closedList);


            //std::cerr << "FN segment: " << segment << "; openList: " << newOpenList.BitsCount() << "; closedList: " << closedList.BitsCount() << std::endl;

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
        std::cerr << "Step: " << result.size() << "; count: " << totalCount << std::endl;
    }

    std::cerr << "Time: " << timer << std::endl;
    return result;
}
