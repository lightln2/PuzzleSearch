#pragma once

#include "../Common/Store.h"

#include <cstdint>
#include <string>
#include <vector>

struct SearchOptions {
    int maxSteps = 1000000;
    int threads = 4;
    std::vector<std::string> directories = { "." };
};


/* three-bit BFS */
template<int size>
std::vector<uint64_t> HanoiSearch(std::string initialState, SearchOptions options);
