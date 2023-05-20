#include "ClassicBFS.h"
#include "../Common/BoolArray.h"

#include <iostream>

std::vector<uint64_t> ClassicBFS(SimpleSlidingPuzzle& puzzle, std::string initialState) {
    const auto SIZE = puzzle.MaxIndexes();
    std::vector<uint64_t> result;

    BoolArray closedList(SIZE);
    BoolArray openList(SIZE);
    BoolArray newOpenList(SIZE);
    auto initialIndex = puzzle.Parse(initialState);
    openList.Set(initialIndex);
    closedList.Set(initialIndex);
    result.push_back(1);

    std::vector<uint64_t> children(puzzle.MaxBranching(), puzzle.INVALID_INDEX);
    std::vector<int> usedOperatorBits(puzzle.OperatorBits(), -1);

    std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {
        uint64_t count = 0;
        openList.ScanBitsAndClear([&](uint64_t index) {
            puzzle.Expand(index, 0, &children[0], &usedOperatorBits[0]);
            for (const auto child : children) {
                if (child == puzzle.INVALID_INDEX) continue;
                if (closedList.Get(child)) continue;
                count++;
                closedList.Set(child);
                newOpenList.Set(child);
            }
        });
        if (count == 0) break;
        result.push_back(count);
        std::swap(openList, newOpenList);
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
    }

    return result;
}


std::vector<uint64_t> TwoBitBFS(SimpleSlidingPuzzle& puzzle, std::string initialState) {

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

    const auto SIZE = puzzle.MaxIndexes();
    std::vector<uint64_t> result;

    int UNVISITED = 0, OLD = 1, CUR = 2, NEXT = 3;

    TwoBitArray array(SIZE);
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
                puzzle.Expand(i, 0, &children[0], &usedOperatorBits[0]);
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

std::vector<uint64_t> ThreeBitBFS(SimpleSlidingPuzzle& puzzle, std::string initialState) {
    const auto SIZE = puzzle.MaxIndexes();
    std::vector<uint64_t> result;

    BoolArray listOld(SIZE);
    BoolArray listCur(SIZE);
    BoolArray listNew(SIZE);

    auto initialIndex = puzzle.Parse(initialState);
    listCur.Set(initialIndex);
    result.push_back(1);

    std::vector<uint64_t> children(puzzle.MaxBranching(), puzzle.INVALID_INDEX);
    std::vector<int> usedOperatorBits(puzzle.OperatorBits(), -1);

    std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {
        uint64_t count = 0;
        listCur.ScanBits([&](uint64_t index) {
            puzzle.Expand(index, 0, &children[0], &usedOperatorBits[0]);
            for (const auto child : children) {
                if (child == puzzle.INVALID_INDEX) continue;
                if (listOld.Get(child)) continue;
                if (listCur.Get(child)) continue;
                if (listNew.Get(child)) continue;
                count++;
                listNew.Set(child);
            }
        });
        if (count == 0) break;
        result.push_back(count);

        std::swap(listOld, listCur);
        std::swap(listCur, listNew);
        listNew.Clear();
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
    }

    return result;
}


std::vector<uint64_t> FrontierSearch(SimpleSlidingPuzzle& puzzle, std::string initialState) {

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
    private:
        std::vector<uint64_t> array;
    };

    const auto SIZE = puzzle.MaxIndexes();
    std::vector<uint64_t> result;

    FourBitArray current(SIZE);
    FourBitArray next(SIZE);

    std::vector<uint64_t> children(puzzle.MaxBranching(), puzzle.INVALID_INDEX);
    std::vector<int> usedOperatorBits(puzzle.OperatorBits(), -1);

    std::cerr << "Step: 0; count: 1" << std::endl;

    while (true) {
        uint64_t count = 0;
        if (result.size() == 0) {
            auto initialIndex = puzzle.Parse(initialState);
            count++;
            puzzle.Expand(initialIndex, 0, &children[0], &usedOperatorBits[0]);
            for (int j = 0; j < children.size(); j++) {
                auto child = children[j];
                if (child == puzzle.INVALID_INDEX) continue;
                next.SetBit(child, usedOperatorBits[j]);
            }
        }
        else {
            for (uint64_t i = 0; i < SIZE; i++) {
                int val = current.Get(i);
                if (val == 0) continue;
                count++;
                puzzle.Expand(i, val, &children[0], &usedOperatorBits[0]);
                for (int j = 0; j < children.size(); j++) {
                    auto child = children[j];
                    if (child == puzzle.INVALID_INDEX) continue;
                    auto bit = usedOperatorBits[j];
                    next.SetBit(child, bit);
                }
            }

        }
        if (count == 0) break;

        std::swap(current, next);
        next.Clear();
        std::cerr << "Step: " << result.size() << "; count: " << count << std::endl;
        result.push_back(count);
    }

    return result;
}
