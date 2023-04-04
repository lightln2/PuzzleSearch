#include "../SlidingTilePuzzle/FrontierSearch.h"

#include <iostream>

int main() {

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
    
    //opts.MaxDepth = 30;

    try {
        FrontierSearch<4, 3>(opts);
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }


}
