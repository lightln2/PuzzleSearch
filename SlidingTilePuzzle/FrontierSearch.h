#pragma once

#include <string>
#include <vector>

struct SearchOptions {
    std::string InitialValue = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15";
    size_t MaxDepth = 1000;
    int Threads = 2;

    std::vector<std::string> FileFrontier1 = { "c:/temp/frontier1" };
    std::vector<std::string> FileFrontier2 = { "c:/temp/frontier2" };
    std::vector<std::string> FileExpandedUp1 = { "c:/temp/expandedUp1" };
    std::vector<std::string> FileExpandedUp2 = { "c:/temp/expandedUp2" };
    std::vector<std::string> FileExpandedDown1 = { "c:/temp/expandedDown1" };
    std::vector<std::string> FileExpandedDown2 = { "c:/temp/expandedDown2" };
};

template<int width, int height>
std::vector<uint64_t> FrontierSearch(SearchOptions options = {});
