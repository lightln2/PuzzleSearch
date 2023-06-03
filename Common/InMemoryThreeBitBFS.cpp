#include "InMemoryBFS.h"
#include "BitArray.h"

std::vector<uint64_t> InMemoryThreeBitBFS(Puzzle& puzzle, std::string initialState) {
    Timer timer;
    const auto SIZE = puzzle.IndexesCount();
    std::vector<uint64_t> result;

    BitArray listOld(SIZE);
    BitArray listCur(SIZE);
    BitArray listNew(SIZE);

    auto initialIndex = puzzle.Parse(initialState);
    listCur.Set(initialIndex);
    result.push_back(1);

    ExpandBuffer nodes(puzzle);
    auto fnExpand = [&](uint64_t child, int op) {
        listNew.Set(child);
    };

    std::cerr << "InMemoryThreeBitBFS" << std::endl;
    std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {

        listCur.ScanBits([&](uint64_t index) {
            nodes.Add(index, 0, fnExpand);
        });
        nodes.Finish(fnExpand);

        if (puzzle.HasOddLengthCycles()) {
            listNew.AndNot(listCur);
        }

        uint64_t count = listNew.AndNotAndCount(listOld);
        if (count == 0) break;
        result.push_back(count);

        std::swap(listOld, listCur);
        std::swap(listCur, listNew);
        listNew.Clear();
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
    }

    std::cerr << "Time: " << timer << std::endl;
    return result;
}
