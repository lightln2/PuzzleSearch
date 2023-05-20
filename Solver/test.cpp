#include "../Common/BoolArray.h"
#include "../Common/Util.h"

#include "../ClassicBFS/GPU.h"
#include "../ClassicBFS/PermutationMap.h"
#include "../ClassicBFS/ClassicBFS.h"

#include "../Common/InMemoryClassicBFS.h"
#include "../Common/InMemoryTwoBitBFS.h"
#include "../Common/InMemoryThreeBitBFS.h"
#include "../Common/InMemoryFrontierSearch.h"
#include "../ClassicBFS/SlidingTilePuzzleSimple.h"

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
    Timer timer;
    std::string initial = "0 1 2 3 4 5 6 7 8 9 10 11";
    /*
    SimpleSlidingPuzzle puzzle(4, 3);
    //ClassicBFS(puzzle, initial);
    ThreeBitBFS(puzzle, initial);
    //FrontierSearch(puzzle, initial);
    */
    SlidingTilePuzzleSimple puzzle(4, 3);
    //InMemoryClassicBFS(puzzle, initial);
    //InMemoryTwoBitBFS(puzzle, initial);
    //InMemoryThreeBitBFS(puzzle, initial);
    InMemoryFrontierSearch(puzzle, initial);
    std::cerr << "Time: " << timer << std::endl;
}