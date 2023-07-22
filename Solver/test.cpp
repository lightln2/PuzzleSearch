#include "../Common/BitArray.h"
#include "../Common/Util.h"

#include "../Common/InMemoryBFS.h"
#include "../ClassicBFS/PancakeOptimized.h"
#include "../ClassicBFS/PancakeOptimizedGpu.h"
#include "../ClassicBFS/PancakeSimple.h"
#include "../ClassicBFS/SlidingTilePuzzleGpu.h"
#include "../ClassicBFS/SlidingTilePuzzleSimple.h"
#include "../ClassicBFS/SlidingTilePuzzleOptimized.h"
#include "../Common/DiskBasedBFS.h"
#include "../ClassicBFS/FourPegHanoiOptimized.h"
#include "../ClassicBFS/FourPegHanoiOptimizedGPU.h"
#include "../ClassicBFS/FourPegHanoiSimple.h"
#include "../ClassicBFS/FourPegHanoiGPU.h"

void TestSlidingTile() {
    //SlidingTilePuzzleSimple puzzle(4, 4);
    //SlidingTilePuzzleGpu puzzle(4, 4);
    SlidingTilePuzzleOptimized<5, 3> puzzle;
    std::string initial = puzzle.ToString(0);

    PuzzleOptions opts;
    //opts.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ", "h:/PUZ2"};
    opts.storeOptions.directories = { "c:/PUZ", "d:/PUZ"};
    opts.segmentBits = 32;
    opts.threads = 2;
    //opts.maxSteps = 25;
    //DiskBasedFrontierSearch(puzzle, initial, opts);
    DiskBasedOptFrontierSearch(puzzle, initial, opts);
    //DiskBasedOptThreeBitBFS(puzzle, initial, opts);
}

void TestDiskBasedHanoi() {
    //FourPegHanoiSimple puzzle(20, true);
    //FourPegHanoiGPU puzzle(19, true);
    //FourPegHanoiOptimized puzzle(19, true);
    FourPegHanoiOptimizedGPU puzzle(22, true);
    std::string initial = puzzle.ToString(0);
    
    PuzzleOptions opts;
    //opts.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ", "h:/PUZ2" };
    opts.storeOptions.directories = { "c:/PUZ", "d:/PUZ"};
    opts.segmentBits = 32;
    opts.threads = 2;
    opts.maxSteps = 140;
    //DiskBasedClassicBFS(puzzle, initial, opts);
    //DiskBasedFrontierSearch(puzzle, initial, opts);
    //DiskBasedOptFrontierSearch(puzzle, initial, opts);
    DiskBasedOptThreeBitBFS(puzzle, initial, opts);
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
