#include "InMemoryClassicBFS.h"
#include "BoolArray.h"

namespace {

    struct FourBitArray {
    public:
        FourBitArray(uint64_t size) : array((size + 15) / 16, 0) {}

        int Get(uint64_t index) {
            auto offset = 4 * (index % 16);
            return (array[index / 16] >> offset) & 15;
        }

        void SetBit(int index, int bit) {
            auto offset = 4 * (index % 16);
            array[index / 16] |= (1ui64 << (offset + bit));
        }

        void Clear() {
            for (uint64_t i = 0; i < array.size(); i++) {
                array[i] = 0;
            }
        }

        template<typename F>
        void ScanAndClear(F func) {
            for (uint64_t i = 0; i < array.size(); i++) {
                uint64_t val = array[i];
                if (val == 0) continue;
                array[i] = 0;
                ScanFourBits(val, i * 16, func);
            }
        }
    private:
        std::vector<uint64_t> array;
    };

} // namespace

std::vector<uint64_t> InMemoryFrontierSearch(Puzzle& puzzle, std::string initialState) {
    const auto SIZE = puzzle.IndexesCount();
    std::vector<uint64_t> result;

    FourBitArray current(SIZE);
    FourBitArray next(SIZE);

    std::vector<uint64_t> indexes;
    std::vector<int> usedOperatorBits;
    std::vector<uint64_t> childIndexes;
    std::vector<int> childOperators;

    indexes.reserve(Puzzle::MAX_INDEXES_BUFFER);
    usedOperatorBits.reserve(Puzzle::MAX_INDEXES_BUFFER);

    std::cerr << "Step: 0; count: 1" << std::endl;

    auto fnExpand = [&]() {
        puzzle.Expand(indexes, usedOperatorBits, childIndexes, childOperators);
        for (int i = 0; i < childIndexes.size(); i++) {
            auto child = childIndexes[i];
            auto bit = childOperators[i];
            if (child == puzzle.INVALID_INDEX) continue;
            next.SetBit(child, bit);
        }

        indexes.clear();
        usedOperatorBits.clear();
        childIndexes.clear();
        childOperators.clear();
    };

    while (true) {
        uint64_t count = 0;
        if (result.size() == 0) {
            auto initialIndex = puzzle.Parse(initialState);
            indexes.push_back(initialIndex);
            usedOperatorBits.push_back(0);
            count++;
            fnExpand();
        }
        else {
            current.ScanAndClear([&](uint64_t index, int val) {
                count++;
                indexes.push_back(index);
                usedOperatorBits.push_back(val);
                if (indexes.size() == Puzzle::MAX_INDEXES_BUFFER) {
                    fnExpand();
                }
            });
            if (indexes.size() > 0) {
                fnExpand();
            }
        }
        if (count == 0) break;

        std::swap(current, next);
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
        result.push_back(count);
    }

    return result;
}
