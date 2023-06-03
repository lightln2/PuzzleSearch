#include "../Common/BitArray.h"
#include "../Common/Util.h"

#include "../Common/InMemoryBFS.h"
#include "../ClassicBFS/SlidingTilePuzzleGpu.h"
#include "../ClassicBFS/SlidingTilePuzzleSimple.h"
#include "../Common/DiskBasedBFS.h"

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

    /*
    {
        SlidingTilePuzzleSimple puzzle(4, 3);
        InMemoryClassicBFS(puzzle, initial);
        InMemoryTwoBitBFS(puzzle, initial);
        InMemoryThreeBitBFS(puzzle, initial);
        InMemoryFrontierSearch(puzzle, initial);
    }
    */
    {
        SlidingTilePuzzleGpu puzzle(4, 3);
        InMemoryClassicBFS(puzzle, initial);
        //InMemoryTwoBitBFS(puzzle, initial);
        //InMemoryThreeBitBFS(puzzle, initial);
        //InMemoryFrontierSearch(puzzle, initial);
    }
}


void TestDiskBasedBFS() {

    std::string initial = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15";

    //SlidingTilePuzzleGpu puzzle(5, 2);
    SlidingTilePuzzleGpu puzzle(7, 2);
    PuzzleOptions opts;
    opts.directories = { "e:/PUZ", "f:/PUZ", "g:/PUZ", "h:/PUZ" };
    //opts.directories = { "c:/temp", "d:/temp"};
    opts.segmentBits = 28;
    opts.threads = 4;
    //opts.maxSteps = 23;
    //opts.segmentBits = 32;
    //DiskBasedClassicBFS(puzzle, initial, opts);
    //DiskBasedThreeBitBFS(puzzle, initial, opts);
    DiskBasedFrontierSearch(puzzle, initial, opts);
    //DiskBasedSinglePassFrontierSearch(puzzle, initial, opts);
}
