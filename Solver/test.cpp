#include "../Common/BitArray.h"
#include "../Common/DiskBasedBFS.h"
#include "../Common/InMemoryBFS.h"
#include "../Common/Util.h"
#include "../Common/FrontierCompression.h"

#include "../Puzzles/PancakeOptimized.h"
#include "../Puzzles/PancakeOptimizedGpu.h"
#include "../Puzzles/PancakeSimple.h"
#include "../Puzzles/SlidingTilePuzzleGpu.h"
#include "../Puzzles/SlidingTilePuzzleSimple.h"
#include "../Puzzles/SlidingTilePuzzleOptimized.h"
#include "../Puzzles/FourPegHanoiOptimized.h"
#include "../Puzzles/FourPegHanoiOptimizedGPU.h"
#include "../Puzzles/FourPegHanoiSimple.h"
#include "../Puzzles/FourPegHanoiGPU.h"

void TestSlidingTile() {
    //SlidingTilePuzzleSimple puzzle(5, 3);
    //SlidingTilePuzzleGpu puzzle(5, 3);
    SlidingTilePuzzleOptimized<5, 3> puzzle;
    std::string initial = puzzle.ToString(0);

    PuzzleOptions opts;
    //opts.storeOptions.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ", "h:/PUZ2"};
    opts.storeOptions.directories = { "c:/PUZ", "d:/PUZ"};
    opts.storeOptions.filesPerPath = 1;
    opts.segmentBits = 32;
    opts.threads = 1;
    opts.maxSteps = 7;
    //DiskBasedFrontierSearch(puzzle, initial, opts);
    DiskBasedOptFrontierSearch(puzzle, initial, opts);
    //DiskBasedOptThreeBitBFS(puzzle, initial, opts);
}

void TestDiskBasedHanoi() {
    FourPegHanoiSimple puzzle(18, true);
    //FourPegHanoiGPU puzzle(17, true);
    //FourPegHanoiOptimized puzzle(17, true);
    //FourPegHanoiOptimizedGPU puzzle(17, true);
    std::string initial = puzzle.ToString(0);
    
    PuzzleOptions opts;
    //opts.storeOptions.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ", "h:/PUZ2" };
    opts.storeOptions.directories = { "c:/PUZ", "d:/PUZ"};
    opts.segmentBits = 32;
    opts.threads = 4;
    opts.maxSteps = 50;
    DiskBasedTwoBitBFS(puzzle, initial, opts);
    //DiskBasedThreeBitBFS(puzzle, initial, opts);
    //DiskBasedOptFrontierSearch(puzzle, initial, opts);
    //DiskBasedOptThreeBitBFS(puzzle, initial, opts);
}

void TestPancake() {
    //PancakeSimple puzzle(13);
    //PancakeOptimized puzzle(13, true);
    PancakeOptimizedGPU puzzle(15, true);

    std::string initial = puzzle.ToString(0);

    PuzzleOptions opts;
    //opts.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ", "h:/PUZ2" };
    opts.storeOptions.directories = { "c:/PUZ", "d:/PUZ"};
    opts.segmentBits = 29;
    opts.threads = 2;
    //opts.maxSteps = 11;
    DiskBasedOptThreeBitBFS2(puzzle, initial, opts);
}

