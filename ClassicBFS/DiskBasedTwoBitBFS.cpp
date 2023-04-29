#include "DiskBasedSearch.h"
#include "../SlidingTilePuzzle/File.h"
#include "../SlidingTilePuzzle/SegmentedFile.h"

#include <string>

namespace {

    struct TwoBitArray {
    public:
        TwoBitArray(uint64_t size) : array((size + 31) / 32, 0) {}

        int Get(uint64_t index) {
            auto offset = 2 * (index % 32);
            return (array[index / 32] >> offset) & 3;
        }

        void Clear(int index) {
            auto offset = 2 * (index % 32);
            array[index / 32] &= ~(3ui64 << offset);
        }

        void Set(int index, int value) {
            auto offset = 2 * (index % 32);
            array[index / 32] &= ~(3ui64 << offset);
            array[index / 32] |= ((uint64_t)value << offset);
        }

        void Clear() {
            for (uint64_t i = 0; i < array.size(); i++) {
                array[i] = 0;
            }
        }

        void Write(SimpleSegmentedFileRW& file, int segment) { 
            file.Delete(segment);
            file.Write(segment, array); 
        }

        void Read(SimpleSegmentedFileRW& file, int segment) { 
            file.Rewind(segment);
            file.Read(segment, array); 
        }
    private:
        std::vector<uint64_t> array;
    };
};

/*
std::vector<uint64_t> DiskBasedTwoBitBFS(SimpleSlidingPuzzle& puzzle, std::string initialState) {
    uint64_t segmentSize = std::min(puzzle.MaxIndexes(), 1ui64 << 32);
    int segmentsCount = (puzzle.MaxIndexes() + segmentSize - 1) / segmentSize;

    SimpleSegmentedFileRW fileArray("./array");
    SimpleSegmentedFileRW crossSegmentCur("./cs1");
    SimpleSegmentedFileRW crossSegmentNew("./cs2");

    TwoBitArray array(segmentSize);

    constexpr int MAXBUF = 10 * 1024 * 1024;
    std::vector<uint32_t> 

    std::vector<uint64_t> result;

    int UNVISITED = 0, OLD = 1, CUR = 2, NEXT = 3;

    for (int i = 0; i < segmentsCount; i++) {
        array.Clear();
        array.Write(fileArray, i);
    }

    for (int i = 0; i < segmentsCount; i++) {

    }

    for (uint64_t i = 0; i < SIZE; i++) array.Set(i, UNVISITED);
    auto initialIndex = puzzle.Parse(initialState);
    array.Set(initialIndex, CUR);
    result.push_back(1);

    std::vector<uint64_t> children(puzzle.MaxBranching(), puzzle.INVALID_INDEX);
    std::vector<int> usedOperatorBits(puzzle.OperatorBits(), -1);

    std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {
        uint64_t count = 0;
        for (uint64_t i = 0; i < SIZE; i++) {
            int val = array.Get(i);
            if (val == CUR) {
                array.Set(i, OLD);
                puzzle.Expand(i, &children[0], &usedOperatorBits[0]);
                for (const auto child : children) {
                    if (child == puzzle.INVALID_INDEX) continue;
                    if (array.Get(child) == UNVISITED) {
                        count++;
                        array.Set(child, NEXT);
                    }
                }
            }
        }
        if (count == 0) break;
        result.push_back(count);

        std::swap(CUR, NEXT);
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
    }

    return result;
}
*/