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

void TestBitArray() {
    constexpr int TRY = 10;
    constexpr uint64_t MAX = 1ui64 << 32;
    constexpr int INC = 13;
    BitArray ba(MAX);
    uint64_t totalNanos = 0;
    for (int t = 0; t < TRY; t++) {
        Timer t1;
        for (uint64_t i = 0; i < MAX; i += INC) {
            ba.Set(i);
        }
        std::cerr << t1 << std::endl;
        Timer t2;
        uint64_t res = 0;
        ba.ScanBitsAndClear([&](uint64_t index) {
            res += index;
        });
        totalNanos += t2.Elapsed();
        ensure(res == 709490155855181350ui64);
        std::cerr << t2 << ": " << res << std::endl;
    }
    std::cerr << "Total: " << WithTime(totalNanos) << std::endl;
}

void TestClassicBFS() {
    std::string initial = "0 1 2 3 4 5 6 7 8 9 10 11";

    SlidingTilePuzzleGpu puzzle(4, 3);
    InMemoryClassicBFS(puzzle, initial);
    //InMemoryTwoBitBFS(puzzle, initial);
    //InMemoryThreeBitBFS(puzzle, initial);
    //InMemoryFrontierSearch(puzzle, initial);
}


void TestDiskBasedBFS() {
    //SlidingTilePuzzleSimple puzzle(4, 4);
    //SlidingTilePuzzleGpu puzzle(4, 4);
    SlidingTilePuzzleOptimized<4, 4> puzzle;
    std::string initial = puzzle.ToString(0);

    PuzzleOptions opts;
    //opts.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ", "h:/PUZ2"};
    opts.directories = { "c:/PUZ", "d:/PUZ"};
    opts.segmentBits = 32;
    opts.threads = 1;
    opts.maxSteps = 25;
    //DiskBasedFrontierSearch(puzzle, initial, opts);
    DiskBasedOptFrontierSearch(puzzle, initial, opts);
    //DiskBasedOptThreeBitBFS(puzzle, initial, opts);
}

void TestDiskBasedHanoi() {
    FourPegHanoiOptimized puzzle(22, true);
    Timer timer;
    ExpandBuffer expander(puzzle);
    uint64_t total = 0;
    auto fn = [&](uint64_t index, int op) {
        total += index;
    };

    for (uint64_t index = 0; index < 100000000; index++) {
        expander.Add(index, 0, fn);
    }
    expander.Finish(fn);
    std::cerr << "Time: " << timer << "; sum: " << WithDecSep(total) << std::endl;

    /*
    //FourPegHanoiSimple puzzle(20, true);
    //FourPegHanoiGPU puzzle(19, true);
    //FourPegHanoiOptimized puzzle(19, true);
    FourPegHanoiOptimizedGPU puzzle(15, true);
    std::string initial = puzzle.ToString(0);

    PuzzleOptions opts;
    //opts.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ", "h:/PUZ2" };
    opts.directories = { "c:/PUZ", "d:/PUZ"};
    opts.segmentBits = 32;
    opts.threads = 2;
    opts.maxSteps = 140;
    //DiskBasedClassicBFS(puzzle, initial, opts);
    //DiskBasedFrontierSearch(puzzle, initial, opts);
    //DiskBasedOptFrontierSearch(puzzle, initial, opts);
    DiskBasedOptThreeBitBFS(puzzle, initial, opts);
    */
}

void TestPancake() {
    //PancakeSimple puzzle(13);
    PancakeOptimized puzzle(13, true);
    //PancakeOptimizedGPU puzzle(15, true);

    std::string initial = puzzle.ToString(0);

    PuzzleOptions opts;
    //opts.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ", "h:/PUZ2" };
    opts.directories = { "c:/PUZ", "d:/PUZ"};
    opts.segmentBits = 29;
    opts.threads = 1;
    //opts.maxSteps = 11;
    DiskBasedOptThreeBitBFS2(puzzle, initial, opts);
}
