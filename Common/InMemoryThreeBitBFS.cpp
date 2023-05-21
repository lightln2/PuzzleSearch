#include "InMemoryClassicBFS.h"
#include "BoolArray.h"

std::vector<uint64_t> InMemoryThreeBitBFS(Puzzle& puzzle, std::string initialState) {
    Timer timer;
    const auto SIZE = puzzle.IndexesCount();
    std::vector<uint64_t> result;

    BoolArray listOld(SIZE);
    BoolArray listCur(SIZE);
    BoolArray listNew(SIZE);

    auto initialIndex = puzzle.Parse(initialState);
    listCur.Set(initialIndex);
    result.push_back(1);

    ExpandBuffer nodes(puzzle);

    std::cerr << "InMemoryThreeBitBFS" << std::endl;
    std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {
        auto fnExpand = [&](uint64_t child, int op) {
            if (listOld.Get(child)) return;
            if (puzzle.HasOddLengthCycles()) {
                if (listCur.Get(child)) return;
            }
            listNew.Set(child);
        };

        listCur.ScanBits([&](uint64_t index) {
            nodes.Add(index, 0, fnExpand);
        });
        nodes.Finish(fnExpand);

        uint64_t count = listNew.BitsCount();
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
