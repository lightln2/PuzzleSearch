#include "pch.h"

#include "../ClassicBFS/SimplePuzzle.h"
#include "../SlidingTilePuzzle/Util.h"

#include <string>
#include <sstream>
#include <vector>

TEST(SimplePuzzleTest, TestSlidingTile3x2) {
    SimpleSlidingPuzzle puz(3, 2);
    auto index = puz.Parse("0 1 2 3 4 5");
    uint64_t children[4];
    int op[4];
    puz.Expand(index, 0, children, op);
    EXPECT_EQ(puz.INVALID_INDEX, children[0]);
    EXPECT_EQ(-1, op[0]);
    EXPECT_EQ("3 1 2 0 4 5", puz.ToString(children[1]));
    EXPECT_NE(-1, op[1]);
    EXPECT_EQ(puz.INVALID_INDEX, children[2]);
    EXPECT_EQ(-1, op[2]);
    EXPECT_EQ("1 0 2 3 4 5", puz.ToString(children[3]));
    EXPECT_NE(-1, op[3]);

}

TEST(SimplePuzzleTest, TestSlidingTile4x4) {
    SimpleSlidingPuzzle puz(4, 4);
    auto index = puz.Parse("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15");
    uint64_t children[4];
    int op[4];
    puz.Expand(index, 0, children, op);
    EXPECT_EQ(puz.INVALID_INDEX, children[0]);
    EXPECT_EQ("4 1 2 3 0 5 6 7 8 9 10 11 12 13 14 15", puz.ToString(children[1]));
    EXPECT_EQ(puz.INVALID_INDEX, children[2]);
    EXPECT_EQ("1 0 2 3 4 5 6 7 8 9 10 11 12 13 14 15", puz.ToString(children[3]));

}

