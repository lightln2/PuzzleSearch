#include "pch.h"

#include "../Puzzles/SlidingTilePuzzleSimple.h"
#include "../Puzzles/SlidingTilePuzzleGpu.h"
#include "../Puzzles/SlidingTilePuzzleOptimized.h"
#include "../Common/Util.h"

#include <string>
#include <sstream>
#include <vector>

TEST(SlidingPuzzleTest, Simple3x2) {
    SlidingTilePuzzleSimple puz(3, 2);
    auto index = puz.Parse("0 1 2 3 4 5");
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("3 1 2 0 4 5", puz.ToString(children[0]));
    EXPECT_NE(-1, ops[0]);
    EXPECT_EQ("1 0 2 3 4 5", puz.ToString(children[1]));
    EXPECT_NE(-1, ops[1]);
}

TEST(SlidingPuzzleTest, Simple4x4) {
    SlidingTilePuzzleSimple puz(4, 4);
    auto index = puz.Parse("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15");
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("4 1 2 3 0 5 6 7 8 9 10 11 12 13 14 15", puz.ToString(children[0]));
    EXPECT_EQ("1 0 2 3 4 5 6 7 8 9 10 11 12 13 14 15", puz.ToString(children[1]));

}

TEST(SlidingPuzzleTest, Gpu3x2) {
    SlidingTilePuzzleGpu puz(3, 2);
    auto index = puz.Parse("0 1 2 3 4 5");
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ(puz.INVALID_INDEX, children[0]);
    EXPECT_EQ(15, ops[0]);
    EXPECT_EQ("3 1 2 0 4 5", puz.ToString(children[1]));
    EXPECT_NE(15, ops[1]);
    EXPECT_EQ(puz.INVALID_INDEX, children[2]);
    EXPECT_EQ(15, ops[2]);
    EXPECT_EQ("1 0 2 3 4 5", puz.ToString(children[3]));
    EXPECT_NE(15, ops[3]);
}

TEST(SlidingPuzzleTest, Gpu4x4) {
    SlidingTilePuzzleGpu puz(4, 4);
    auto index = puz.Parse("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15");
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ(puz.INVALID_INDEX, children[0]);
    EXPECT_EQ("4 1 2 3 0 5 6 7 8 9 10 11 12 13 14 15", puz.ToString(children[1]));
    EXPECT_EQ(puz.INVALID_INDEX, children[2]);
    EXPECT_EQ("1 0 2 3 4 5 6 7 8 9 10 11 12 13 14 15", puz.ToString(children[3]));

}

TEST(SlidingPuzzleTest, Opt3x2) {
    SlidingTilePuzzleOptimized<3, 2> puz;
    auto index = puz.Parse("0 1 2 3 4 5");
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ(puz.INVALID_INDEX, children[0]);
    EXPECT_EQ(15, ops[0]);
    EXPECT_EQ("3 1 2 0 4 5", puz.ToString(children[1]));
    EXPECT_NE(-1, ops[1]);
    EXPECT_EQ(puz.INVALID_INDEX, children[2]);
    EXPECT_EQ(15, ops[2]);
    EXPECT_EQ("1 0 2 3 4 5", puz.ToString(children[3]));
    EXPECT_NE(-1, ops[3]);
}

TEST(SlidingPuzzleTest, Opt4x4) {
    SlidingTilePuzzleOptimized<4, 4> puz;
    auto index = puz.Parse("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15");
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ(puz.INVALID_INDEX, children[0]);
    EXPECT_EQ("4 1 2 3 0 5 6 7 8 9 10 11 12 13 14 15", puz.ToString(children[1]));
    EXPECT_EQ(puz.INVALID_INDEX, children[2]);
    EXPECT_EQ("1 0 2 3 4 5 6 7 8 9 10 11 12 13 14 15", puz.ToString(children[3]));

}

