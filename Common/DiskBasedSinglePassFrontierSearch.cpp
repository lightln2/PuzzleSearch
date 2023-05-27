#include "BoolArray.h"
#include "DiskBasedBFS.h"
#include "SegmentReader.h"
#include "SegmentWriter.h"
#include "Multiplexor.h"
#include "Store.h"
#include "Util.h"

std::vector<uint64_t> DiskBasedSinglePassFrontierSearch(Puzzle& puzzle, std::string initialState, PuzzleOptions opts) {
    std::cerr << "DB_FrontierSearch" << std::endl;
    Timer timer;
    const int OPS = puzzle.OperatorsCount();
    const int OPS_MASK = (1 << OPS) - 1;
    const uint64_t SIZE = puzzle.IndexesCount();
    uint64_t SEGMENT_SIZE = 1ui64 << opts.segmentBits;
    const uint64_t SEGMENT_MASK = SEGMENT_SIZE - 1;
    const int SEGMENTS = (SIZE + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
    if (SEGMENTS == 1 && SEGMENT_SIZE > SIZE) {
        SEGMENT_SIZE = SIZE; // SEGMENT_MASK is still valid
    }

    // classic FrontierSearch stores indexes and used operator bits in a single 4-byte word
    ensure(opts.segmentBits + OPS <= 32);

    std::cerr
        << "total: " << WithDecSep(SIZE)
        << "; segments: " << WithDecSep(SEGMENTS)
        << "; segment size: " << WithDecSep(SEGMENT_SIZE) << std::endl;

    std::vector<uint64_t> result;

    std::vector<std::string> frontierDirs1;
    std::vector<std::string> frontierDirs2;
    std::vector<std::string> crossSegmentDirs1;
    std::vector<std::string> crossSegmentDirs2;
    for (const auto& dir : opts.directories) {
        frontierDirs1.push_back(dir + "/f1/");
        frontierDirs2.push_back(dir + "/f2/");
        crossSegmentDirs1.push_back(dir + "/xseg1/");
        crossSegmentDirs2.push_back(dir + "/xseg2/");
    }

    Store curFrontierStore = Store::CreateMultiFileStore(SEGMENTS, frontierDirs1);
    Store newFrontierStore = Store::CreateMultiFileStore(SEGMENTS, frontierDirs2);
    Store currentCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, crossSegmentDirs1);
    Store nextCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, crossSegmentDirs2);

    SegmentReader currentXSegReader(currentCrossSegmentStore);
    Multiplexor mult(nextCrossSegmentStore, SEGMENTS);
    SegmentReader frontierReader(curFrontierStore);
    SegmentWriter frontierWriter(newFrontierStore);

    MultiBitArray array(OPS, SEGMENT_SIZE);
    BoolArray frontierArray(SEGMENT_SIZE);

    auto fnGetSegIdx = [&](uint64_t index) {
        return std::pair<int, uint32_t>(index >> opts.segmentBits, index & SEGMENT_MASK);
    };

    auto fnGetIndexAndOp = [&](uint32_t value) {
        return std::pair<uint32_t, int>(value >> OPS, value & OPS_MASK);
    };

    auto fnGetValue = [&](uint32_t index, int op) {
        return (index << OPS) | op;
    };

    ExpandBuffer nodes(puzzle);

    auto initialIndex = puzzle.Parse(initialState);
    auto [seg, idx] = fnGetSegIdx(initialIndex);
    frontierWriter.SetSegment(seg);
    frontierWriter.Add(fnGetValue(idx, 0));
    frontierWriter.Flush();
    //Expand cross-segment!!!
    {
        auto fnExpandCrossSegment = [&](uint64_t child, int op) {
            auto [s, idx] = fnGetSegIdx(child);
            if (s == seg) return;
            mult.Add(s, fnGetValue(idx, op));
        };

        nodes.Add(initialIndex, 0, fnExpandCrossSegment);
        nodes.Finish(fnExpandCrossSegment);
        mult.FlushAllSegments();
    }
    std::swap(curFrontierStore, newFrontierStore);
    std::swap(currentCrossSegmentStore, nextCrossSegmentStore);
    result.push_back(1);

    uint64_t total_sz_frontier = 0;
    uint64_t total_sz_xseg = 0;

    std::cerr << "Step: 1; Count: 1" << std::endl;

    while (true) {
        Timer stepTimer;
        uint64_t totalCount = 0;

        for (int segment = 0; segment < SEGMENTS; segment++) {
            uint64_t indexBase = (uint64_t)segment << opts.segmentBits;

            bool hasData = false;

            currentXSegReader.SetSegment(segment);
            frontierReader.SetSegment(segment);
            frontierWriter.SetSegment(segment);

            while (true) {
                auto& vect = currentXSegReader.Read();
                if (vect.empty()) break;
                hasData = true;
                for (uint32_t val : vect) {
                    auto [idx, op] = fnGetIndexAndOp(val);
                    array.Set(idx, op);
                }
            }
            currentCrossSegmentStore.Delete(segment);

            auto fnExpandInSegment = [&](uint64_t child, int op) {
                auto [seg, idx] = fnGetSegIdx(child);
                if (seg != segment) return;
                array.Set(idx, op);
            };

            while (true) {
                auto& vect = frontierReader.Read();
                if (vect.empty()) break;
                hasData = true;
                for (uint32_t val : vect) {
                    auto [idx, op] = fnGetIndexAndOp(val);
                    nodes.Add(indexBase | idx, op, fnExpandInSegment);
                    if (puzzle.HasOddLengthCycles()) frontierArray.Set(idx);
                }
            }

            if (!hasData) {
                continue;
            }

            nodes.Finish(fnExpandInSegment);
            curFrontierStore.Delete(segment);

            auto fnExpandCrossSegment = [&](uint64_t child, int op) {
                auto [seg, idx] = fnGetSegIdx(child);
                if (seg == segment) return;
                mult.Add(seg, fnGetValue(idx, op));
            };

            uint64_t count = 0;
            array.ScanBitsAndClear([&](uint64_t index, int opBits) {
                if (puzzle.HasOddLengthCycles() && frontierArray.Get(index)) return;
                count++;
                nodes.Add(indexBase | index, opBits, fnExpandCrossSegment);
                frontierWriter.Add(fnGetValue(index, opBits));
            });
            nodes.Finish(fnExpandCrossSegment);
            frontierWriter.Flush();

            if (puzzle.HasOddLengthCycles()) frontierArray.Clear();

            totalCount += count;
        }

        mult.FlushAllSegments();

        if (totalCount == 0) break;
        result.push_back(totalCount);
        curFrontierStore.DeleteAll();
        currentCrossSegmentStore.DeleteAll();
        std::swap(curFrontierStore, newFrontierStore);
        std::swap(currentCrossSegmentStore, nextCrossSegmentStore);
        std::cerr
            << "Step: " << result.size()
            << "; count: " << totalCount
            << " in " << stepTimer
            << "; size: frontier=" << WithSize(curFrontierStore.TotalLength())
            << ", x-seg=" << WithSize(currentCrossSegmentStore.TotalLength())
            << std::endl;
        total_sz_frontier += curFrontierStore.TotalLength();
        total_sz_xseg += currentCrossSegmentStore.TotalLength();
    }

    std::cerr << "Time: " << timer << std::endl;
    Store::PrintStats();
    std::cerr
        << "Total files: frontier=" << WithSize(total_sz_frontier)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    return result;
}
