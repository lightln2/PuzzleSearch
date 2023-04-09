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
    opts.Threads = 5;

    /*
    opts.FileFrontier1 = { "e:/PUZ/frontier1", "h:/PUZ/frontier1" };
    opts.FileFrontier2 = { "h:/PUZ/frontier2", "f:/PUZ/frontier2" };
    opts.FileExpandedUp1 = { "g:/PUZ/expandedUp1" };
    opts.FileExpandedDown1 = { "g:/PUZ/expandedDown1" };
    opts.FileExpandedUp2 = { "g:/PUZ/expandedUp2" };
    opts.FileExpandedDown2 = { "g:/PUZ/expandedDown2" };

    opts.FileFrontierVert1 = { "e:/PUZ/frontier-vert-1", "f:/PUZ/frontier-vert-1" };
    opts.FileFrontierHoriz1 = { "f:/PUZ/frontier-horiz-1", "e:/PUZ/frontier-horiz-1" };
    opts.FileFrontierVert2 = { "h:/PUZ/frontier-vert-2" };
    opts.FileFrontierHoriz2 = { "h:/PUZ/frontier-horiz-2" };
    opts.FileExpanded1 = { "g:/PUZ/frontierExp1.part1", "g:/PUZ/frontierExp1.part2", "g:/PUZ/frontierExp1.part3", "g:/PUZ/frontierExp1.part4", "g:/PUZ/frontierExp1.part5" };
    opts.FileExpanded2 = { "g:/PUZ/frontierExp2.part1", "g:/PUZ/frontierExp2.part2", "g:/PUZ/frontierExp2.part3", "g:/PUZ/frontierExp2.part4", "g:/PUZ/frontierExp2.part5" };
    opts.ExpandedFileSequentialParts = true;
    */

    opts.FileFrontierVert1 = { "d:/PUZ/frontier-vert-1" };
    opts.FileFrontierHoriz1 = { "d:/PUZ/frontier-horiz-1" };
    opts.FileFrontierVert2 = { "d:/PUZ/frontier-vert-2" };
    opts.FileFrontierHoriz2 = { "d:/PUZ/frontier-horiz-2" };
    opts.FileExpanded1 = { "c:/PUZ/frontierExp1.part1", "c:/PUZ/frontierExp1.part2", "c:/PUZ/frontierExp1.part3", "c:/PUZ/frontierExp1.part4", "c:/PUZ/frontierExp1.part5" };
    opts.FileExpanded2 = { "c:/PUZ/frontierExp2.part1", "c:/PUZ/frontierExp2.part2", "c:/PUZ/frontierExp2.part3", "c:/PUZ/frontierExp2.part4", "c:/PUZ/frontierExp2.part5" };
    opts.ExpandedFileSequentialParts = true;

    //opts.MaxDepth = 17;

    try {
        MTFrontierSearch<7, 2>(opts);
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }


}

int main() {
    MTFrontierSearch();
}
