#include "InMemoryBFS.h"
#include "BitArray.h"

#include <optional>

namespace {

    struct FourBitArray {
    public:
        FourBitArray(uint64_t size) : array((size + 15) / 16, 0) {}

        int Get(uint64_t index) {
            auto offset = 4 * (index % 16);
            return (array[index / 16] >> offset) & 15;
        }

        void SetBit(uint64_t index, int bit) {
            auto offset = 4 * (index % 16);
            array[index / 16] |= (1ui64 << (offset + bit));
        }

        void Clear(uint64_t index) {
            auto offset = 4 * (index % 16);
            array[index / 16] &= ~(15ui64 << offset);
        }

        void Clear() {
            for (uint64_t i = 0; i < array.size(); i++) {
                array[i] = 0;
            }
        }

        template<typename F>
        void Scan(F func) {
            for (uint64_t i = 0; i < array.size(); i++) {
                uint64_t val = array[i];
                if (val == 0) continue;
                ScanFourBits(val, i * 16, func);
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
    Timer timer;
    const auto SIZE = puzzle.IndexesCount();
    std::vector<uint64_t> result;

    FourBitArray current(SIZE);
    FourBitArray next(SIZE);

    BitArray currentExclude(puzzle.HasOddLengthCycles() ? SIZE : 0);
    BitArray nextExclude(puzzle.HasOddLengthCycles() ? SIZE : 0);

    ExpandBuffer nodes(puzzle);

    auto fnExpand = [&](uint64_t child, int op) {
        next.SetBit(child, op);
    };

    std::cerr << "InMemoryFrontierSearch" << std::endl;

    while (true) {
        Timer timerStep;

        uint64_t count = 0;

        if (result.size() == 0) {
            auto initialIndex = puzzle.Parse(initialState);
            nodes.Add(initialIndex, 0, fnExpand);
            if (puzzle.HasOddLengthCycles()) {
                nextExclude.Set(initialIndex);
            }
            count++;
        }
        else {
            current.ScanAndClear([&](uint64_t index, int val) {
                if (puzzle.HasOddLengthCycles()) {
                    if (currentExclude.Get(index)) return;
                    nextExclude.Set(index);
                }
                nodes.Add(index, val, fnExpand);
                count++;
            });
        }
        nodes.Finish(fnExpand);

        if (count == 0) break;

        std::swap(current, next);
        std::swap(currentExclude, nextExclude);
        nextExclude.Clear();
        std::cerr << "Step: " << result.size() << "; count: " << count << " in " << timerStep << std::endl;
        result.push_back(count);
    }

    std::cerr << "Time: " << timer << std::endl;
    return result;
}
