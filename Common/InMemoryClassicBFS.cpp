#include "InMemoryClassicBFS.h"
#include "BoolArray.h"

std::vector<uint64_t> InMemoryClassicBFS(Puzzle& puzzle, std::string initialState) {
    Timer timer;
    const auto SIZE = puzzle.IndexesCount();
    std::vector<uint64_t> result;

    BoolArray closedList(SIZE);
    BoolArray openList(SIZE);
    BoolArray newOpenList(SIZE);
    auto initialIndex = puzzle.Parse(initialState);
    openList.Set(initialIndex);
    closedList.Set(initialIndex);
    result.push_back(1);

    ExpandBuffer nodes(puzzle);

    std::cerr << "InMemoryClassicBFS" << std::endl;
    std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {
        uint64_t count = 0;

        auto fnExpand = [&](uint64_t child, int op) {
            if (closedList.Get(child)) return;
            count++;
            closedList.Set(child);
            newOpenList.Set(child);
        };

        openList.ScanBitsAndClear([&](uint64_t index) {
            nodes.Add(index, 0, fnExpand);
        });
        nodes.Finish(fnExpand);

        if (count == 0) break;
        result.push_back(count);
        std::swap(openList, newOpenList);
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
    }

    std::cerr << "Time: " << timer << std::endl;
    return result;
}
