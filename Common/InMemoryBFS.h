#pragma once

#include "Puzzle.h"

std::vector<uint64_t> InMemoryClassicBFS(Puzzle& puzzle, std::string initialState);

std::vector<uint64_t> InMemoryTwoBitBFS(Puzzle& puzzle, std::string initialState);

std::vector<uint64_t> InMemoryThreeBitBFS(Puzzle& puzzle, std::string initialState);

std::vector<uint64_t> InMemoryFrontierSearch(Puzzle& puzzle, std::string initialState);
