#include "../SlidingTilePuzzle/FrontierSearch.h"
#include "../SlidingTilePuzzle/MTFrontierSearch.h"

#include <iostream>

void FrontierSearch() {

    SearchOptions opts;

    opts.Threads = 4;

    /*
    opts.FileFrontier1 = { "e:/PUZ/frontier1", "h:/PUZ/frontier1" };
    opts.FileFrontier2 = { "h:/PUZ/frontier2", "f:/PUZ/frontier2" };
    opts.FileExpandedUp1 = { "g:/PUZ/expandedUp1" };
    opts.FileExpandedDown1 = { "g:/PUZ/expandedDown1" };
    opts.FileExpandedUp2 = { "g:/PUZ/expandedUp2" };
    opts.FileExpandedDown2 = { "g:/PUZ/expandedDown2" };
    */

    opts.FileFrontier1 = { "d:/PUZ/frontier1" };
    opts.FileFrontier2 = { "d:/PUZ/frontier2" };
    opts.FileExpandedUp1 = { "c:/PUZ/frontierUp1" };
    opts.FileExpandedDown1 = { "c:/PUZ/frontierDown1" };
    opts.FileExpandedUp2 = { "c:/PUZ/expandedUp2" };
    opts.FileExpandedDown2 = { "c:/PUZ/expandedDown2" };

    //opts.MaxDepth = 35;

    try {
        FrontierSearch<4, 3>(opts);
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }


}

void MTFrontierSearch() {

    MTSearchOptions opts;
    opts.Threads = 1;

    /*
    opts.FileFrontier1 = { "e:/PUZ/frontier1", "h:/PUZ/frontier1" };
    opts.FileFrontier2 = { "h:/PUZ/frontier2", "f:/PUZ/frontier2" };
    opts.FileExpandedUp1 = { "g:/PUZ/expandedUp1" };
    opts.FileExpandedDown1 = { "g:/PUZ/expandedDown1" };
    opts.FileExpandedUp2 = { "g:/PUZ/expandedUp2" };
    opts.FileExpandedDown2 = { "g:/PUZ/expandedDown2" };
    */

    opts.FileFrontierVert1 = { "d:/PUZ/frontier-vert-1" };
    opts.FileFrontierHoriz1 = { "d:/PUZ/frontier-horiz-1" };
    opts.FileFrontierVert2 = { "d:/PUZ/frontier-vert-2" };
    opts.FileFrontierHoriz2 = { "d:/PUZ/frontier-horiz-2" };
    opts.FileExpanded1 = { "c:/PUZ/frontierExp1" };
    opts.FileExpanded2 = { "c:/PUZ/expandedExp2" };

    //opts.MaxDepth = 10;

    try {
        MTFrontierSearch<6, 2>(opts);
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }


}

int main() {
    MTFrontierSearch();
}
