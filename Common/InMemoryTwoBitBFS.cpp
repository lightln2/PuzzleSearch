#include "InMemoryBFS.h"
#include "BitArray.h"

namespace {

    struct TwoBitArray {
    public:
        TwoBitArray(uint64_t size) : array(size * 2, false) {}

        int Get(uint64_t index) {
            return array[2 * index] * 2 + array[2 * index + 1];
        }

        void Set(uint64_t index, int value) {
            array[2 * index] = value / 2;
            array[2 * index + 1] = value % 2;
        }
    private:
        std::vector<bool> array;
    };

} // namespace

std::vector<uint64_t> InMemoryTwoBitBFS(Puzzle& puzzle, std::string initialState) {
    Timer timer;
    const auto SIZE = puzzle.IndexesCount();
    std::vector<uint64_t> result;

    int UNVISITED = 0, OLD = 1, CUR = 2, NEXT = 3;

    ExpandBuffer nodes(puzzle);

    TwoBitArray array(SIZE);
    for (uint64_t i = 0; i < SIZE; i++) array.Set(i, UNVISITED);
    auto initialIndex = puzzle.Parse(initialState);
    array.Set(initialIndex, CUR);
    result.push_back(1);

    std::cerr << "InMemoryTwoBitBFS" << std::endl;
    std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {
        uint64_t count = 0;

        auto fnExpand = [&](uint64_t child, int op) {
            if (array.Get(child) == UNVISITED) {
                count++;
                array.Set(child, NEXT);
            }
        };

        for (uint64_t i = 0; i < SIZE; i++) {
            int val = array.Get(i);
            if (val != CUR) continue;
            array.Set(i, OLD);
            nodes.Add(i, 0, fnExpand);
        }
        nodes.Finish(fnExpand);

        if (count == 0) break;
        result.push_back(count);

        std::swap(CUR, NEXT);
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
    }

    std::cerr << "Time: " << timer << std::endl;
    return result;
}
