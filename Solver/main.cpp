#include "../SlidingTilePuzzle/FrontierSearch.h"
#include "../SlidingTilePuzzle/MTFrontierSearch.h"

#include "../SlidingTilePuzzle/SegmentedFile.h"
#include "../SlidingTilePuzzle/Collector.h"

#include <iostream>

void FrontierSearch4x3() {
    FrontierSearch<4, 3>();
}

void SlidingTileFrontierSearch();

void MTFrontierSearch() {

    MTSearchOptions opts;
    opts.Threads = 5;


    opts.FileFrontierVert1 = { "e:/PUZ/fv1", "f:/PUZ/fv1", "h:/PUZ/fv1.p1", "h:/PUZ/fv1.p2" };
    opts.FileFrontierHoriz1 = { "f:/PUZ/fh1", "h:/PUZ/fh1.p1", "h:/PUZ/fh1.p2", "e:/PUZ/fh1" };
    opts.FileFrontierVert2 = { "h:/PUZ/fv2.p1", "h:/PUZ/fv2.p2", "e:/PUZ/fv2", "f:/PUZ/fv2" };
    opts.FileFrontierHoriz2 = { "h:/PUZ/fh2.p2", "e:/PUZ/fh2", "f:/PUZ/fh2", "h:/PUZ/fh2.p1" };

    opts.FileExpanded1 = { "g:/PUZ/e1.part1", "g:/PUZ/e1.part2", "g:/PUZ/e1.part3", "g:/PUZ/e1.part4", "g:/PUZ/e1.part5" };
    opts.FileExpanded2 = { "g:/PUZ/e2.part1", "g:/PUZ/e2.part2", "g:/PUZ/e2.part3", "g:/PUZ/e2.part4", "g:/PUZ/e2.part5" };
    opts.ExpandedFileSequentialParts = true;

    //opts.FileExpanded1 = { "h:/PUZ/e2.part1", "h:/PUZ/e2.part2", "e:/PUZ/e2.part3", "f:/PUZ/e2.part4" };
    //opts.FileExpanded2 = { "e:/PUZ/e1.part1", "f:/PUZ/e1.part2", "h:/PUZ/e1.part3", "h:/PUZ/e1.part4" };
    //opts.FileSmallExpanded1 = "g:/PUZ/se1";
    //opts.FileSmallExpanded2 = "g:/PUZ/se2";
    //opts.SmallFileLimit = 2 * 1024 * 1024;

    opts.InitialValue = "1 2 3 4 5  6 7 0 8 9  10 11 12 13 14";


    /*
    opts.FileFrontierVert1 = { "d:/PUZ/fv1" };
    opts.FileFrontierHoriz1 = { "d:/PUZ/fh1" };
    opts.FileFrontierVert2 = { "d:/PUZ/fv2" };
    opts.FileFrontierHoriz2 = { "d:/PUZ/fh2" };
    opts.FileExpanded1 = { "d:/PUZ/e1" };
    opts.FileExpanded2 = { "d:/PUZ/e2" };
    opts.FileSmallExpanded1 = "c:/PUZ/se1";
    opts.FileSmallExpanded2 = "c:/PUZ/se2";
    opts.SmallFileLimit = 4 * 1024 * 1024;
    //opts.MaxDepth = 20;
    */


    MTFrontierSearch<5, 3>(opts);
}

void MTFrontierSearch4x3() {
    MTFrontierSearch<4, 3>();
}

void HanoiSearch();

void TestSlidingTile();
void TestDiskBasedHanoi();
void TestPancake();


int main() {
    try {
        //FrontierSearch4x3();
        SlidingTileFrontierSearch();
        //MTFrontierSearch4x3();
        //MTFrontierSearch();

        //HanoiSearch();

        //TestSlidingTile();
        //TestDiskBasedHanoi();
        //TestPancake();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        throw;
    }
}
