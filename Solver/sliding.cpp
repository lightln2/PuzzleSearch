#include "../SlidingTilePuzzle/FrontierSearch.h"

void SlidingTileFrontierSearch() {

    STSearchOptions opts;

    opts.Threads = 4;


    opts.FileFrontier1 = { "e:/PUZ/f1", "f:/PUZ/f1", "h:/PUZ/f1.p1", "h:/PUZ/f1.p2" };
    opts.FileFrontier2 = { "e:/PUZ/f2", "f:/PUZ/f2", "h:/PUZ/f2.p1", "h:/PUZ/f2.p2" };
    opts.FileExpandedUp1 = { "g:/PUZ/expandedUp1" };
    opts.FileExpandedDown1 = { "g:/PUZ/expandedDown1" };
    opts.FileExpandedUp2 = { "g:/PUZ/expandedUp2" };
    opts.FileExpandedDown2 = { "g:/PUZ/expandedDown2" };

    /*
    opts.FileFrontier1 = { "d:/PUZ/frontier1" };
    opts.FileFrontier2 = { "d:/PUZ/frontier2" };
    opts.FileExpandedUp1 = { "c:/PUZ/frontierUp1" };
    opts.FileExpandedDown1 = { "c:/PUZ/frontierDown1" };
    opts.FileExpandedUp2 = { "c:/PUZ/expandedUp2" };
    opts.FileExpandedDown2 = { "c:/PUZ/expandedDown2" };
    */

    opts.InitialValue = "0 1 2 3 4  5 6 7 8 9  10 11 12 13 14";

    //opts.MaxDepth = 35;

    FrontierSearch<5, 3>(opts);
}
