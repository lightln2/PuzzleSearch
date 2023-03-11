#pragma once

#include <string>
#include <vector>

struct SearchOptions {
    std::string InitialValue = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15";
    size_t MaxDepth = 1000;
};

template<int width, int height>
std::vector<uint64_t> FrontierSearch(SearchOptions options = {});