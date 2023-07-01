#include "pch.h"

#include "../ClassicBFS/PancakeSimple.h"
#include "../Common/Util.h"

#include <string>
#include <sstream>
#include <vector>

TEST(PancakeTest, Pancake5) {
    PancakeSimple puz(5);
    auto index = 0;
    auto stateStr = puz.ToString(index);
    EXPECT_EQ(stateStr, "0 1 2 3 4");
    EXPECT_EQ(puz.Parse(stateStr), 0);
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("1 0 2 3 4", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("2 1 0 3 4", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("3 2 1 0 4", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("4 3 2 1 0", puz.ToString(children[3]));
    EXPECT_EQ(3, ops[3]);
}

