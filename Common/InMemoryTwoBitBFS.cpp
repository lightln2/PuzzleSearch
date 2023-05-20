#include "InMemoryClassicBFS.h"
#include "BoolArray.h"

namespace {

    struct TwoBitArray {
    public:
        TwoBitArray(uint64_t size) : array(size * 2, false) {}

        int Get(int index) {
            return array[2 * index] * 2 + array[2 * index + 1];
        }

        void Set(int index, int value) {
            array[2 * index] = value / 2;
            array[2 * index + 1] = value % 2;
        }
    private:
        std::vector<bool> array;
    };

} // namespace

std::vector<uint64_t> InMemoryTwoBitBFS(Puzzle& puzzle, std::string initialState) {

    const auto SIZE = puzzle.IndexesCount();
    std::vector<uint64_t> result;

    int UNVISITED = 0, OLD = 1, CUR = 2, NEXT = 3;

    std::vector<uint64_t> indexes;
    std::vector<int> usedOperatorBits;
    std::vector<uint64_t> childIndexes;
    std::vector<int> childOperators;

    indexes.reserve(Puzzle::MAX_INDEXES_BUFFER);
    usedOperatorBits.reserve(Puzzle::MAX_INDEXES_BUFFER);

    TwoBitArray array(SIZE);
    for (uint64_t i = 0; i < SIZE; i++) array.Set(i, UNVISITED);
    auto initialIndex = puzzle.Parse(initialState);
    array.Set(initialIndex, CUR);
    result.push_back(1);

    auto fnExpand = [&]() {
        int cnt = 0;
        puzzle.Expand(indexes, usedOperatorBits, childIndexes, childOperators);
        for (const auto child : childIndexes) {
            if (child == puzzle.INVALID_INDEX) continue;
            if (array.Get(child) == UNVISITED) {
                cnt++;
                array.Set(child, NEXT);
            }
        }
        indexes.clear();
        usedOperatorBits.clear();
        childIndexes.clear();
        childOperators.clear();
        return cnt;
    };

    std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {
        uint64_t count = 0;
        for (uint64_t i = 0; i < SIZE; i++) {
            int val = array.Get(i);
            if (val != CUR) continue;
            array.Set(i, OLD);
            indexes.push_back(i);
            usedOperatorBits.push_back(0);
            if (indexes.size() == Puzzle::MAX_INDEXES_BUFFER) {
                count += fnExpand();
            }
        }
        if (indexes.size() > 0) {
            count += fnExpand();
        }
        if (count == 0) break;
        result.push_back(count);

        std::swap(CUR, NEXT);
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
    }

    return result;
}
