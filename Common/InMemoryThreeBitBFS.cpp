#include "InMemoryClassicBFS.h"
#include "BoolArray.h"

std::vector<uint64_t> InMemoryThreeBitBFS(Puzzle& puzzle, std::string initialState) {
    const auto SIZE = puzzle.IndexesCount();
    std::vector<uint64_t> result;

    BoolArray listOld(SIZE);
    BoolArray listCur(SIZE);
    BoolArray listNew(SIZE);

    auto initialIndex = puzzle.Parse(initialState);
    listCur.Set(initialIndex);
    result.push_back(1);

    std::vector<uint64_t> indexes;
    std::vector<int> usedOperatorBits;
    std::vector<uint64_t> childIndexes;
    std::vector<int> childOperators;

    indexes.reserve(Puzzle::MAX_INDEXES_BUFFER);
    usedOperatorBits.reserve(Puzzle::MAX_INDEXES_BUFFER);

    std::cerr << "Step: 0; count: 1" << std::endl;

    auto fnExpand = [&]() {
        int cnt = 0;
        puzzle.Expand(indexes, usedOperatorBits, childIndexes, childOperators);
        for (const auto child : childIndexes) {
            if (child == puzzle.INVALID_INDEX) continue;
            if (listOld.Get(child)) continue;
            if (listCur.Get(child)) continue;
            if (listNew.Get(child)) continue;
            cnt++;
            listNew.Set(child);
        }
        indexes.clear();
        usedOperatorBits.clear();
        childIndexes.clear();
        childOperators.clear();
        return cnt;
    };

    while (true) {
        uint64_t count = 0;
        listCur.ScanBits([&](uint64_t index) {
            indexes.push_back(index);
            usedOperatorBits.push_back(0);
            if (indexes.size() == Puzzle::MAX_INDEXES_BUFFER) {
                count += fnExpand();
            }
        });
        if (indexes.size() > 0) {
            count += fnExpand();
        }

        if (count == 0) break;
        result.push_back(count);

        std::swap(listOld, listCur);
        std::swap(listCur, listNew);
        listNew.Clear();
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
    }

    return result;
}
