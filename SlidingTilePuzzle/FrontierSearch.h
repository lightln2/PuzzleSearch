#pragma once

#include <string>
#include <vector>

struct SearchOptions {
    std::string InitialValue = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15";
    size_t MaxDepth = 1000;
    int Threads = 2;

    std::string FileFrontier1 = "c:/temp/frontier1";
    std::string FileFrontier2 = "c:/temp/frontier2";
    std::string FileFrontierCS1 = "c:/temp/frontierCS1";
    std::string FileFrontierCS2 = "c:/temp/frontierCS2";
    std::string FileExpanded1 = "c:/temp/expanded1";
    std::string FileExpanded2 = "c:/temp/expanded2";
};

template<int width, int height>
std::vector<uint64_t> FrontierSearch(SearchOptions options = {});
