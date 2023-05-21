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