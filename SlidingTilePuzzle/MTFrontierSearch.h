#pragma once

#include <string>
#include <vector>

struct MTSearchOptions {
    std::string InitialValue = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15";
    size_t MaxDepth = 1000;
    int Threads = 2;

    std::vector<std::string> FileFrontierHoriz1 = { "c:/temp/frontier_h1" };
    std::vector<std::string> FileFrontierVert1 = { "c:/temp/frontier_v1" };
    std::vector<std::string> FileFrontierHoriz2 = { "c:/temp/frontier_h2" };
    std::vector<std::string> FileFrontierVert2 = { "c:/temp/frontier_v2" };
    std::vector<std::string> FileExpanded1 = { "c:/temp/expanded1" };
    std::vector<std::string> FileExpanded2 = { "c:/temp/expanded2" };
};

template<int width, int height>
std::vector<uint64_t> MTFrontierSearch(MTSearchOptions options = {});
