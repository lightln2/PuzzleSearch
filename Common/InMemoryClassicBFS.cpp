#include "InMemoryBFS.h"
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

    auto fnExpand = [&](uint64_t child, int op) {
        newOpenList.Set(child);
    };


    std::cerr << "InMemoryClassicBFS" << std::endl;
    std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {
        openList.ScanBitsAndClear([&](uint64_t index) {
            nodes.Add(index, 0, fnExpand);
        });
        nodes.Finish(fnExpand);

        uint64_t count = newOpenList.AndNotAndCount(closedList);
        closedList.Or(newOpenList);

        if (count == 0) break;
        result.push_back(count);
        std::swap(openList, newOpenList);
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
    }

    std::cerr << "Time: " << timer << std::endl;
    ExpandBuffer::PrintStats();
    return result;
}
