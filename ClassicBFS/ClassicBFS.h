#pragma once

#include "SimplePuzzle.h"

#include <cstdint>
#include <vector>

std::vector<uint64_t> ClassicBFS(SimpleSlidingPuzzle& puzzle, std::string initialState);

std::vector<uint64_t> TwoBitBFS(SimpleSlidingPuzzle& puzzle, std::string initialState);

std::vector<uint64_t> ThreeBitBFS(SimpleSlidingPuzzle& puzzle, std::string initialState);

std::vector<uint64_t> FrontierSearch(SimpleSlidingPuzzle& puzzle, std::string initialState);
