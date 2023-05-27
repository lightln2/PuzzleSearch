#pragma once

#include "Puzzle.h"

#include <string>
#include <vector>

struct PuzzleOptions {
    size_t maxSteps = 10000;
    int segmentBits = 32;
    std::vector<std::string> directories = { "." };
};

std::vector<uint64_t> DiskBasedClassicBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts = {});
std::vector<uint64_t> DiskBasedTwoBitBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts = {});
std::vector<uint64_t> DiskBasedThreeBitBFS(Puzzle& puzzle, std::string initialState, PuzzleOptions opts = {});
std::vector<uint64_t> DiskBasedFrontierSearch(Puzzle& puzzle, std::string initialState, PuzzleOptions opts = {});
std::vector<uint64_t> DiskBasedSinglePassFrontierSearch(Puzzle& puzzle, std::string initialState, PuzzleOptions opts = {});
