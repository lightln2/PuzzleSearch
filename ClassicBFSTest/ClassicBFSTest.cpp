#include "pch.h"

#include "../ClassicBFS/ClassicBFS.h"
#include "../SlidingTilePuzzle/Util.h"

#include <string>
#include <sstream>
#include <vector>

static std::string ToString(const std::vector<uint64_t>& result) {
    std::ostringstream stream;
    for (int i = 0; i < result.size(); i++) {
        if (i > 0) stream << ' ';
        stream << result[i];
    }
    return stream.str();
}

TEST(ClassicBFSTest, Classic_SlidingTile3x2) {
    SimpleSlidingPuzzle puzzle(3, 2);
    auto result = ClassicBFS(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), "1 2 3 5 6 7 10 12 12 16 23 25 28 39 44 40 29 21 18 12 6 1");
}

TEST(ClassicBFSTest, Classic_SlidingTile4x2) {
    SimpleSlidingPuzzle puzzle(4, 2);
    auto result = ClassicBFS(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), "1 2 3 6 10 14 19 28 42 61 85 119 161 215 293 396 506 632 788 985 1194 1414 1664 1884 1999 1958 1770 1463 1076 667 361 190 88 39 19 7 1");
}

TEST(ClassicBFSTest, TwoBit_SlidingTile3x2) {
    SimpleSlidingPuzzle puzzle(3, 2);
    auto result = TwoBitBFS(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), "1 2 3 5 6 7 10 12 12 16 23 25 28 39 44 40 29 21 18 12 6 1");
}

TEST(ClassicBFSTest, TwoBit_SlidingTile4x2) {
    SimpleSlidingPuzzle puzzle(4, 2);
    auto result = TwoBitBFS(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), "1 2 3 6 10 14 19 28 42 61 85 119 161 215 293 396 506 632 788 985 1194 1414 1664 1884 1999 1958 1770 1463 1076 667 361 190 88 39 19 7 1");
}


TEST(ClassicBFSTest, ThreeBit_SlidingTile3x2) {
    SimpleSlidingPuzzle puzzle(3, 2);
    auto result = ThreeBitBFS(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), "1 2 3 5 6 7 10 12 12 16 23 25 28 39 44 40 29 21 18 12 6 1");
}

TEST(ClassicBFSTest, ThreeBit_SlidingTile4x2) {
    SimpleSlidingPuzzle puzzle(4, 2);
    auto result = ThreeBitBFS(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), "1 2 3 6 10 14 19 28 42 61 85 119 161 215 293 396 506 632 788 985 1194 1414 1664 1884 1999 1958 1770 1463 1076 667 361 190 88 39 19 7 1");
}

TEST(ClassicBFSTest, Frontier_SlidingTile3x2) {
    SimpleSlidingPuzzle puzzle(3, 2);
    auto result = FrontierSearch(puzzle, "0 1 2 3 4 5");
    EXPECT_EQ(ToString(result), "1 2 3 5 6 7 10 12 12 16 23 25 28 39 44 40 29 21 18 12 6 1");
}

TEST(ClassicBFSTest, Frontier_SlidingTile4x2) {
    SimpleSlidingPuzzle puzzle(4, 2);
    auto result = FrontierSearch(puzzle, "0 1 2 3 4 5 6 7");
    EXPECT_EQ(ToString(result), "1 2 3 6 10 14 19 28 42 61 85 119 161 215 293 396 506 632 788 985 1194 1414 1664 1884 1999 1958 1770 1463 1076 667 361 190 88 39 19 7 1");
}