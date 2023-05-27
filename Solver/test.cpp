#include "../Common/BoolArray.h"
#include "../Common/Util.h"

#include "../Common/InMemoryBFS.h"
#include "../ClassicBFS/SlidingTilePuzzleGpu.h"
#include "../ClassicBFS/SlidingTilePuzzleSimple.h"
#include "../Common/DiskBasedBFS.h"

void TestBoolArray() {
    constexpr int TRY = 10;
    constexpr uint64_t MAX = 1ui64 << 32;
    constexpr int INC = 13;
    BoolArray ba(MAX);
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
        //InMemoryClassicBFS(puzzle, initial);
        //InMemoryTwoBitBFS(puzzle, initial);
        InMemoryThreeBitBFS(puzzle, initial);
        //InMemoryFrontierSearch(puzzle, initial);
    }
}


void TestDiskBasedBFS() {
    std::string initial = "0 1 2 3 4 5 6 7 8 9 10 11 12 13";

    //SlidingTilePuzzleGpu puzzle(3, 3);
    SlidingTilePuzzleGpu puzzle(4, 3);
    PuzzleOptions opts;
    opts.directories = { "c:/temp" };
    //opts.segmentBits = 28;
    opts.segmentBits = 32;

    DiskBasedThreeBitBFS(puzzle, initial, opts);
}
