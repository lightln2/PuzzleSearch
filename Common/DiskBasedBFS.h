#pragma once

#include "Puzzle.h"
#include "Store.h"

#include <string>
#include <vector>

struct PuzzleOptions {
    size_t maxSteps = 10000;
    int segmentBits = 32;
    std::vector<std::string> directories = { "." };
    int threads = 1;
};

struct SegmentedOptions {
    SegmentedOptions(Puzzle& puzzle, PuzzleOptions& opts);
    Store MakeStore(std::string suffix);
    StoreSet MakeStoreSet(std::string suffix, int count);
    void PrintOptions();

    __forceinline std::pair<int, uint32_t> GetSegIdx(uint64_t index) {
        return { int(index >> Opts.segmentBits), uint32_t(index & SegmentMask) };
    };

    Puzzle& Puzzle;
    PuzzleOptions Opts;
    uint64_t TotalSize;
    int Segments;
    uint64_t SegmentSize;
    uint64_t SegmentMask;
    int OperatorsCount;
    int OperatorsMask;
    bool HasOddLengthCycles;
};

std::vector<uint64_t> DiskBasedClassicBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts = {});
std::vector<uint64_t> DiskBasedTwoBitBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts = {});
std::vector<uint64_t> DiskBasedThreeBitBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts = {});
std::vector<uint64_t> DiskBasedFrontierSearch(Puzzle& puzzle, std::string initialState, PuzzleOptions opts = {});
std::vector<uint64_t> DiskBasedOptFrontierSearch(Puzzle& puzzle, std::string initialState, PuzzleOptions opts = {});
