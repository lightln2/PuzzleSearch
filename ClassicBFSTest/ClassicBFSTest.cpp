#include "pch.h"

#include "../ClassicBFS/SlidingTilePuzzleGpu.h"
#include "../ClassicBFS/SlidingTilePuzzleSimple.h"
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

TEST(SlidingTile_DiskBased_GPU, SPFS_5x2_1seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 22;
    auto result = DiskBasedSinglePassFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, SPFS_5x2_7seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    auto result = DiskBasedSinglePassFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, SPFS_5x2_7seg_4th) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    opts.threads = 4;
    auto result = DiskBasedSinglePassFrontierSearch(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, FrontierSearch2_5x2_1seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 22;
    auto result = DiskBasedFrontierSearch2(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, FrontierSearch2_5x2_7seg) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    auto result = DiskBasedFrontierSearch2(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}

TEST(SlidingTile_DiskBased_GPU, FrontierSearch2_5x2_7seg_4th) {
    SlidingTilePuzzleGpu puzzle(5, 2);
    PuzzleOptions opts;
    opts.segmentBits = 19;
    opts.threads = 4;
    auto result = DiskBasedFrontierSearch2(puzzle, "0 1 2 3 4 5 6 7 8 9", opts);
    EXPECT_EQ(ToString(result), ST5x2);
}
