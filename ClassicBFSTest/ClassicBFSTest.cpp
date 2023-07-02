#include "pch.h"

#include "../ClassicBFS/PancakeSimple.h"
#include "../ClassicBFS/PancakeOptimized.h"
#include "../ClassicBFS/PancakeOptimizedGpu.h"
#include "../ClassicBFS/SlidingTilePuzzleGpu.h"
#include "../ClassicBFS/SlidingTilePuzzleOptimized.h"
#include "../ClassicBFS/SlidingTilePuzzleSimple.h"
#include "../ClassicBFS/FourPegHanoiSimple.h"
#include "../ClassicBFS/FourPegHanoiGPU.h"
#include "../Common/InMemoryBFS.h"
#include "../Common/DiskBasedBFS.h"

#include <string>
#include <sstream>
#include <vector>

static std::string ST3x2 = "1 2 3 5 6 7 10 12 12 16 23 25 28 39 44 40 29 21 18 12 6 1";
static std::string ST4x2 = "1 2 3 6 10 14 19 28 42 61 85 119 161 215 293 396 506 632 788 985 1194 1414 1664 1884 1999 1958 1770 1463 1076 667 361 190 88 39 19 7 1";
static std::string ST5x2 = "1 2 3 6 11 19 30 44 68 112 176 271 411 602 851 1232 1783 2530 3567 4996 6838 9279 12463 16597 21848 28227 35682 44464 54597 65966 78433 91725 104896 116966 126335 131998 133107 128720 119332 106335 91545 75742 60119 45840 33422 23223 15140 9094 5073 2605 1224 528 225 75 20 2";

static std::string ToString(const std::vector<uint64_t>& result) {
    std::ostringstream stream;
    for (int i = 0; i < result.size(); i++) {
        if (i > 0) stream << ' ';
        stream << result[i];
    }
    return stream.str();
}

// CPU

TEST(SlidingTile_CPU, ClassicBFS_3x2) {
    SlidingTilePuzzleSimple puzzle(3, 2);
    auto result = InMemoryClassicBFS(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), ST3x2);
}

TEST(SlidingTile_CPU, ClassicBFS_4x2) {
    SlidingTilePuzzleSimple puzzle(4, 2);
    auto result = InMemoryClassicBFS(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), ST4x2);
}

TEST(SlidingTile_CPU, TwoBitBFS_3x2) {
    SlidingTilePuzzleSimple puzzle(3, 2);
    auto result = InMemoryTwoBitBFS(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), ST3x2);
}

TEST(SlidingTile_CPU, TwoBitBFS_4x2) {
    SlidingTilePuzzleSimple puzzle(4, 2);
    auto result = InMemoryTwoBitBFS(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), ST4x2);
}

TEST(SlidingTile_CPU, ThreeBitBFS_3x2) {
    SlidingTilePuzzleSimple puzzle(3, 2);
    auto result = InMemoryThreeBitBFS(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), ST3x2);
}

TEST(SlidingTile_CPU, ThreeBitBFS_4x2) {
    SlidingTilePuzzleSimple puzzle(4, 2);
    auto result = InMemoryThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), ST4x2);
}

TEST(SlidingTile_CPU, FrontierSearch_3x2) {
    SlidingTilePuzzleSimple puzzle(3, 2);
    auto result = InMemoryFrontierSearch(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), ST3x2);
}

TEST(SlidingTile_CPU, FrontierSearch_4x2) {
    SlidingTilePuzzleSimple puzzle(4, 2);
    auto result = InMemoryFrontierSearch(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), ST4x2);
}

// GPU

TEST(SlidingTile_GPU, ClassicBFS_3x2) {
    SlidingTilePuzzleGpu puzzle(3, 2);
    auto result = InMemoryClassicBFS(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), ST3x2);
}

TEST(SlidingTile_GPU, ClassicBFS_4x2) {
    SlidingTilePuzzleGpu puzzle(4, 2);
    auto result = InMemoryClassicBFS(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), ST4x2);
}

TEST(SlidingTile_GPU, TwoBitBFS_3x2) {
    SlidingTilePuzzleGpu puzzle(3, 2);
    auto result = InMemoryTwoBitBFS(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), ST3x2);
}

TEST(SlidingTile_GPU, TwoBitBFS_4x2) {
    SlidingTilePuzzleGpu puzzle(4, 2);
    auto result = InMemoryTwoBitBFS(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), ST4x2);
}

TEST(SlidingTile_GPU, ThreeBitBFS_3x2) {
    SlidingTilePuzzleGpu puzzle(3, 2);
    auto result = InMemoryThreeBitBFS(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), ST3x2);
}

TEST(SlidingTile_GPU, ThreeBitBFS_4x2) {
    SlidingTilePuzzleGpu puzzle(4, 2);
    auto result = InMemoryThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), ST4x2);
}

TEST(SlidingTile_GPU, FrontierSearch_3x2) {
    SlidingTilePuzzleGpu puzzle(3, 2);
    auto result = InMemoryFrontierSearch(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), ST3x2);
}

TEST(SlidingTile_GPU, FrontierSearch_4x2) {
    SlidingTilePuzzleGpu puzzle(4, 2);
    auto result = InMemoryFrontierSearch(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), ST4x2);
}

TEST(SlidingTile_DiskBased_GPU, ClassicBFS_5x2_1seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 22;
    auto result = DiskBasedClassicBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, ClassicBFS_5x2_7seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    auto result = DiskBasedClassicBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, ClassicBFS_5x2_7seg_4th) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    opts.threads = 4;
    auto result = DiskBasedClassicBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, TwoBitBFS_5x2_1seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 22;
    auto result = DiskBasedTwoBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, TwoBitBFS_5x2_7seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    auto result = DiskBasedTwoBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, TwoBitBFS_5x2_7seg_4th) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    opts.threads = 4;
    auto result = DiskBasedTwoBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, ThreeBitBFS_5x2_1seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 22;
    auto result = DiskBasedThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, ThreeBitBFS_5x2_7seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    auto result = DiskBasedThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, ThreeBitBFS_5x2_7seg_4th) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    opts.threads = 4;
    auto result = DiskBasedThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, FrontierSearch_5x2_1seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 22;
    auto result = DiskBasedFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, FrontierSearch_5x2_7seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    auto result = DiskBasedFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, FrontierSearch_5x2_7seg_4th) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    opts.threads = 4;
    auto result = DiskBasedFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, OptFrontierSearch_5x2_1seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 22;
    auto result = DiskBasedOptFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, OptFrontierSearch_5x2_7seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    auto result = DiskBasedOptFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, OptFrontierSearch_5x2_7seg_4th) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    opts.threads = 4;
    auto result = DiskBasedOptFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, Opt3BitBFS_5x2_1seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 22;
    auto result = DiskBasedOptThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, Opt3BitBFS_5x2_7seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    auto result = DiskBasedOptThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, Opt3BitBFS_5x2_7seg_4th) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    opts.threads = 4;
    auto result = DiskBasedOptThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}



TEST(SlidingTile_Optimized_GPU, OptFrontierSearch_5x2_1seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 22;
    auto result = DiskBasedOptFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_Optimized_GPU, OptFrontierSearch_5x2_7seg) {
    SlidingTilePuzzleOptimized<5, 2> puzzle;
    PuzzleOptions opts;
    opts.segmentBits = 19;
    auto result = DiskBasedOptFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_Optimized_GPU, OptFrontierSearch_5x2_7seg_4th) {
    SlidingTilePuzzleOptimized<5, 2> puzzle;
    PuzzleOptions opts;
    opts.segmentBits = 19;
    opts.threads = 4;
    auto result = DiskBasedOptFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_Optimized_GPU, Opt3BitBFS_5x2_1seg) {
    SlidingTilePuzzleOptimized<5, 2> puzzle;
    PuzzleOptions opts;
    opts.segmentBits = 22;
    auto result = DiskBasedOptThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_Optimized_GPU, Opt3BitBFS_5x2_7seg) {
    SlidingTilePuzzleOptimized<5, 2> puzzle;
    PuzzleOptions opts;
    opts.segmentBits = 19;
    auto result = DiskBasedOptThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_Optimized_GPU, Opt3BitBFS_5x2_7seg_4th) {
    SlidingTilePuzzleOptimized<5, 2> puzzle;
    PuzzleOptions opts;
    opts.segmentBits = 19;
    opts.threads = 4;
    auto result = DiskBasedOptThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}



TEST(HanoiTowers_CPU, ClassicBFS_10) {
    FourPegHanoiSimple puzzle(10);
    PuzzleOptions opts;
    opts.segmentBits = 32;
    opts.threads = 1;
    auto result = DiskBasedClassicBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 3 6 12 30 30 66 96 126 210 330 318 462 816 1032 936 1044 1752 2610 3036 3528 3306 4578 6318 9108 10674 11580 11844 13374 17124 23664 32184 36984 39810 38484 39768 45498 56838 74880 91506 106134 109890 91878 63528 45474 32598 12978 1908 210 12");
}

TEST(HanoiTowers_CPU, OptFS_10_mt) {
    FourPegHanoiSimple puzzle(10);
    PuzzleOptions opts;
    opts.segmentBits = 20;
    opts.threads = 4;
    auto result = DiskBasedOptFrontierSearch(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 3 6 12 30 30 66 96 126 210 330 318 462 816 1032 936 1044 1752 2610 3036 3528 3306 4578 6318 9108 10674 11580 11844 13374 17124 23664 32184 36984 39810 38484 39768 45498 56838 74880 91506 106134 109890 91878 63528 45474 32598 12978 1908 210 12");
}

TEST(HanoiTowers_CPU, ClassicBFS_10_Sym) {
    FourPegHanoiSimple puzzle(10, true);
    PuzzleOptions opts;
    opts.segmentBits = 32;
    opts.threads = 1;
    auto result = DiskBasedClassicBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 1 1 3 5 7 11 16 24 36 55 56 81 137 172 162 183 293 435 506 588 562 779 1056 1519 1780 1930 1983 2249 2871 3957 5367 6165 6638 6425 6654 7608 9511 12500 15255 17690 18320 15339 10630 7614 5496 2214 334 52 4");
}

TEST(HanoiTowers_CPU, OptFS_10_mt_Sym) {
    FourPegHanoiSimple puzzle(10, true);
    PuzzleOptions opts;
    opts.segmentBits = 20;
    opts.threads = 4;
    auto result = DiskBasedOptFrontierSearch(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 1 1 3 5 7 11 16 24 36 55 56 81 137 172 162 183 293 435 506 588 562 779 1056 1519 1780 1930 1983 2249 2871 3957 5367 6165 6638 6425 6654 7608 9511 12500 15255 17690 18320 15339 10630 7614 5496 2214 334 52 4");
}

TEST(HanoiTowers_CPU, Opt3BitBFS_10_mt_Sym) {
    FourPegHanoiSimple puzzle(10, true);
    PuzzleOptions opts;
    opts.segmentBits = 20;
    opts.threads = 4;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 1 1 3 5 7 11 16 24 36 55 56 81 137 172 162 183 293 435 506 588 562 779 1056 1519 1780 1930 1983 2249 2871 3957 5367 6165 6638 6425 6654 7608 9511 12500 15255 17690 18320 15339 10630 7614 5496 2214 334 52 4");
}

TEST(HanoiTowers_GPU, ClassicBFS_10_Sym) {
    FourPegHanoiGPU puzzle(10, true);
    PuzzleOptions opts;
    opts.segmentBits = 32;
    opts.threads = 1;
    auto result = DiskBasedClassicBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 1 1 3 5 7 11 16 24 36 55 56 81 137 172 162 183 293 435 506 588 562 779 1056 1519 1780 1930 1983 2249 2871 3957 5367 6165 6638 6425 6654 7608 9511 12500 15255 17690 18320 15339 10630 7614 5496 2214 334 52 4");
}

TEST(HanoiTowers_GPU, OptFS_10_mt_Sym) {
    FourPegHanoiGPU puzzle(10, true);
    PuzzleOptions opts;
    opts.segmentBits = 20;
    opts.threads = 4;
    auto result = DiskBasedOptFrontierSearch(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 1 1 3 5 7 11 16 24 36 55 56 81 137 172 162 183 293 435 506 588 562 779 1056 1519 1780 1930 1983 2249 2871 3957 5367 6165 6638 6425 6654 7608 9511 12500 15255 17690 18320 15339 10630 7614 5496 2214 334 52 4");
}

TEST(HanoiTowers_GPU, Opt3BitBFS_10_mt_Sym) {
    FourPegHanoiGPU puzzle(10, true);
    PuzzleOptions opts;
    opts.segmentBits = 20;
    opts.threads = 4;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 1 1 3 5 7 11 16 24 36 55 56 81 137 172 162 183 293 435 506 588 562 779 1056 1519 1780 1930 1983 2249 2871 3957 5367 6165 6638 6425 6654 7608 9511 12500 15255 17690 18320 15339 10630 7614 5496 2214 334 52 4");
}

TEST(HanoiTowers_GPU, Opt3BitBFS_21) {
    FourPegHanoiGPU puzzle(21, false);
    PuzzleOptions opts;
    opts.segmentBits = 32;
    opts.threads = 2;
    opts.maxSteps = 10;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 3 6 12 30 30 66 96 126 210 330");
}

TEST(HanoiTowers_GPU, Opt3BitBFS_21_Sym) {
    FourPegHanoiGPU puzzle(21, true);
    PuzzleOptions opts;
    opts.segmentBits = 32;
    opts.threads = 2;
    opts.maxSteps = 10;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 1 1 3 5 7 11 16 24 36 55");
}

TEST(Pancake_CPU, Opt3BitBFS_5) {
    PancakeSimple puzzle(5);
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0));
    EXPECT_EQ(ToString(result), "1 4 12 35 48 20");
}

TEST(Pancake_CPU, Opt3BitBFS_9) {
    PancakeSimple puzzle(9);
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0));
    EXPECT_EQ(ToString(result), "1 8 56 391 2278 10666 38015 93585 132697 79379 5804");
}

TEST(Pancake_CPU, Opt3BitBFS_9_mt) {
    PancakeSimple puzzle(9);
    PuzzleOptions opts;
    opts.segmentBits = 15;
    opts.threads = 3;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 8 56 391 2278 10666 38015 93585 132697 79379 5804");
}

TEST(Pancake_CPU, Opt3BitBFS_13_mt) {
    PancakeSimple puzzle(13);
    PuzzleOptions opts;
    opts.segmentBits = 29;
    opts.threads = 3;
    opts.maxSteps = 6;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 12 132 1451 14556 130096 1030505");
}

TEST(Pancake_Opt_CPU, Opt3BitBFS_13_mt) {
    PancakeOptimized puzzle(13);
    PuzzleOptions opts;
    opts.segmentBits = 29;
    opts.threads = 3;
    opts.maxSteps = 6;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 12 132 1451 14556 130096 1030505");
}

TEST(Pancake_Opt_CPU, Opt3BitBFS_13_mt_invIdx) {
    PancakeOptimized puzzle(13, true);
    PuzzleOptions opts;
    opts.segmentBits = 29;
    opts.threads = 3;
    opts.maxSteps = 6;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 12 132 1451 14556 130096 1030505");
}

TEST(Pancake_Opt_GPU, Opt3BitBFS_5) {
    PancakeOptimizedGPU puzzle(5, false);
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0));
    EXPECT_EQ(ToString(result), "1 4 12 35 48 20");
}

TEST(Pancake_Opt_GPU, Opt3BitBFS_9) {
    PancakeOptimizedGPU puzzle(9, false);
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0));
    EXPECT_EQ(ToString(result), "1 8 56 391 2278 10666 38015 93585 132697 79379 5804");
}

TEST(Pancake_Opt_GPU, Opt3BitBFS_9_mt) {
    PancakeOptimizedGPU puzzle(9, false);
    PuzzleOptions opts;
    opts.segmentBits = 15;
    opts.threads = 3;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 8 56 391 2278 10666 38015 93585 132697 79379 5804");
}

TEST(Pancake_Opt_GPU, Opt3BitBFS_13_mt) {
    PancakeOptimizedGPU puzzle(13, false);
    PuzzleOptions opts;
    opts.segmentBits = 29;
    opts.threads = 3;
    opts.maxSteps = 6;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 12 132 1451 14556 130096 1030505");
}

TEST(Pancake_Opt_GPU, Opt3BitBFS_13_mt_invIdx) {
    PancakeOptimizedGPU puzzle(13, true);
    PuzzleOptions opts;
    opts.segmentBits = 29;
    opts.threads = 3;
    opts.maxSteps = 6;
    auto result = DiskBasedOptThreeBitBFS(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 12 132 1451 14556 130096 1030505");
}

TEST(Pancake_Opt_GPU, Opt3BitBFS2_9_mt) {
    PancakeOptimizedGPU puzzle(9, false);
    PuzzleOptions opts;
    opts.segmentBits = 15;
    opts.threads = 3;
    auto result = DiskBasedOptThreeBitBFS2(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 8 56 391 2278 10666 38015 93585 132697 79379 5804");
}

TEST(Pancake_Opt_GPU, Opt3BitBFS2_13_mt) {
    PancakeOptimizedGPU puzzle(13, false);
    PuzzleOptions opts;
    opts.segmentBits = 29;
    opts.threads = 3;
    opts.maxSteps = 6;
    auto result = DiskBasedOptThreeBitBFS2(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 12 132 1451 14556 130096 1030505");
}

TEST(Pancake_Opt_GPU, Opt3BitBFS2_13_mt_invIdx) {
    PancakeOptimizedGPU puzzle(13, true);
    PuzzleOptions opts;
    opts.segmentBits = 29;
    opts.threads = 3;
    opts.maxSteps = 6;
    auto result = DiskBasedOptThreeBitBFS2(puzzle, puzzle.ToString(0), opts);
    EXPECT_EQ(ToString(result), "1 12 132 1451 14556 130096 1030505");
}

