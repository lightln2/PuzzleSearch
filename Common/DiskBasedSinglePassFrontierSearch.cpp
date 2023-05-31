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
    const int SEGMENTS = int((SIZE + SEGMENT_SIZE - 1) / SEGMENT_SIZE);
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

    Store curFrontierStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "frontier1");
    Store newFrontierStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "frontier2");
    Store currentCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "xseg1");
    Store nextCrossSegmentStore = Store::CreateMultiFileStore(SEGMENTS, opts.directories, "xseg2");

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
    uint64_t nanos_collect = 0;

    std::cerr << "Step: 0; Count: 1" << std::endl;

    while (result.size() <= opts.maxSteps) {
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
                if (vect.IsEmpty()) break;
                hasData = true;
                for (size_t i = 0; i < vect.Size(); i++) {
                    uint32_t val = vect[i];
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
                if (vect.IsEmpty()) break;
                hasData = true;
                for (size_t i = 0; i < vect.Size(); i++) {
                    uint32_t val = vect[i];
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
            Timer timerCollect;
            array.ScanBitsAndClear([&](uint64_t index, int opBits) {
                if (puzzle.HasOddLengthCycles() && frontierArray.Get(index)) return;
                count++;
                if (opBits < OPS_MASK) {
                    nodes.Add(indexBase | index, opBits, fnExpandCrossSegment);
                    frontierWriter.Add(fnGetValue(uint32_t(index), opBits));
                }
            });
            nodes.Finish(fnExpandCrossSegment);
            frontierWriter.Flush();
            nanos_collect += timerCollect.Elapsed();

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
            << "Step: " << result.size() - 1
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
    ExpandBuffer::PrintStats();
    std::cerr << "Collect: " << WithTime(nanos_collect) << std::endl;
    std::cerr
        << "Total files: frontier=" << WithSize(total_sz_frontier)
        << "; x-seg=" << WithSize(total_sz_xseg)
        << std::endl;
    return result;
}
