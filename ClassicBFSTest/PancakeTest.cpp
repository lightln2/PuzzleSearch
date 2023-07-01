#include "pch.h"

#include "../ClassicBFS/PancakeOptimized.h"
#include "../ClassicBFS/PancakeOptimizedGpu.h"
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
    EXPECT_EQ("0 1 2 4 3", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("0 1 4 3 2", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("0 4 3 2 1", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("4 3 2 1 0", puz.ToString(children[3]));
    EXPECT_EQ(3, ops[3]);
}

TEST(PancakeTest, OptPancake5) {
    PancakeOptimized puz(5);
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
    EXPECT_EQ("0 1 2 4 3", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("0 1 4 3 2", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("0 4 3 2 1", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("4 3 2 1 0", puz.ToString(children[3]));
    EXPECT_EQ(3, ops[3]);
}

TEST(PancakeTest, OptPancake15) {
    PancakeOptimized puz(15);
    auto index = 0;
    auto stateStr = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14";
    EXPECT_EQ(puz.Parse(stateStr), index);
    EXPECT_EQ(puz.ToString(index), stateStr);
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 11 12 14 13", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 11 14 13 12", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 14 13 12 11", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 14 13 12 11 10", puz.ToString(children[3]));
    EXPECT_EQ(3, ops[3]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 14 13 12 11 10 9", puz.ToString(children[4]));
    EXPECT_EQ(4, ops[4]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 14 13 12 11 10 9 8", puz.ToString(children[5]));
    EXPECT_EQ(5, ops[5]);
    EXPECT_EQ("0 1 2 3 4 5 6 14 13 12 11 10 9 8 7", puz.ToString(children[6]));
    EXPECT_EQ(6, ops[6]);
    EXPECT_EQ("0 1 2 3 4 5 14 13 12 11 10 9 8 7 6", puz.ToString(children[7]));
    EXPECT_EQ(7, ops[7]);
    EXPECT_EQ("0 1 2 3 4 14 13 12 11 10 9 8 7 6 5", puz.ToString(children[8]));
    EXPECT_EQ(8, ops[8]);
    EXPECT_EQ("0 1 2 3 14 13 12 11 10 9 8 7 6 5 4", puz.ToString(children[9]));
    EXPECT_EQ(9, ops[9]);
    EXPECT_EQ("0 1 2 14 13 12 11 10 9 8 7 6 5 4 3", puz.ToString(children[10]));
    EXPECT_EQ(10, ops[10]);

    EXPECT_EQ("0 1 14 13 12 11 10 9 8 7 6 5 4 3 2", puz.ToString(children[11]));
    EXPECT_EQ(11, ops[11]);
    EXPECT_EQ("0 14 13 12 11 10 9 8 7 6 5 4 3 2 1", puz.ToString(children[12]));
    EXPECT_EQ(12, ops[12]);
    EXPECT_EQ("14 13 12 11 10 9 8 7 6 5 4 3 2 1 0", puz.ToString(children[13]));
    EXPECT_EQ(13, ops[13]);

    EXPECT_EQ(children[0] >> 29, 0);
    EXPECT_EQ(children[1] >> 29, 0);
    EXPECT_EQ(children[2] >> 29, 0);
    EXPECT_EQ(children[3] >> 29, 0);
    EXPECT_EQ(children[4] >> 29, 0);
    EXPECT_EQ(children[5] >> 29, 0);
    EXPECT_EQ(children[6] >> 29, 0);
    EXPECT_EQ(children[7] >> 29, 0);
    EXPECT_EQ(children[8] >> 29, 0);
    EXPECT_EQ(children[9] >> 29, 0);
    EXPECT_EQ(children[10] >> 29, 0);

    EXPECT_EQ(children[11] >> 29, 12);
    EXPECT_EQ(children[12] >> 29, 181);
    EXPECT_EQ(children[13] >> 29, 2729);
}

TEST(PancakeTest, OptPancake15Inv) {
    PancakeOptimized puz(15, true);
    auto index = 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 - 1;
    auto stateStr = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14";
    EXPECT_EQ(puz.Parse(stateStr), index);
    EXPECT_EQ(puz.ToString(index), stateStr);
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 11 12 14 13", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 11 14 13 12", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 14 13 12 11", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 14 13 12 11 10", puz.ToString(children[3]));
    EXPECT_EQ(3, ops[3]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 14 13 12 11 10 9", puz.ToString(children[4]));
    EXPECT_EQ(4, ops[4]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 14 13 12 11 10 9 8", puz.ToString(children[5]));
    EXPECT_EQ(5, ops[5]);
    EXPECT_EQ("0 1 2 3 4 5 6 14 13 12 11 10 9 8 7", puz.ToString(children[6]));
    EXPECT_EQ(6, ops[6]);
    EXPECT_EQ("0 1 2 3 4 5 14 13 12 11 10 9 8 7 6", puz.ToString(children[7]));
    EXPECT_EQ(7, ops[7]);
    EXPECT_EQ("0 1 2 3 4 14 13 12 11 10 9 8 7 6 5", puz.ToString(children[8]));
    EXPECT_EQ(8, ops[8]);
    EXPECT_EQ("0 1 2 3 14 13 12 11 10 9 8 7 6 5 4", puz.ToString(children[9]));
    EXPECT_EQ(9, ops[9]);
    EXPECT_EQ("0 1 2 14 13 12 11 10 9 8 7 6 5 4 3", puz.ToString(children[10]));
    EXPECT_EQ(10, ops[10]);

    EXPECT_EQ("0 1 14 13 12 11 10 9 8 7 6 5 4 3 2", puz.ToString(children[11]));
    EXPECT_EQ(11, ops[11]);
    EXPECT_EQ("0 14 13 12 11 10 9 8 7 6 5 4 3 2 1", puz.ToString(children[12]));
    EXPECT_EQ(12, ops[12]);
    EXPECT_EQ("14 13 12 11 10 9 8 7 6 5 4 3 2 1 0", puz.ToString(children[13]));
    EXPECT_EQ(13, ops[13]);

    EXPECT_EQ(children[0] >> 29, 0);
    EXPECT_EQ(children[1] >> 29, 0);
    EXPECT_EQ(children[2] >> 29, 0);
    EXPECT_EQ(children[3] >> 29, 0);
    EXPECT_EQ(children[4] >> 29, 0);
    EXPECT_EQ(children[5] >> 29, 0);
    EXPECT_EQ(children[6] >> 29, 0);
    EXPECT_EQ(children[7] >> 29, 0);
    EXPECT_EQ(children[8] >> 29, 0);
    EXPECT_EQ(children[9] >> 29, 0);
    EXPECT_EQ(children[10] >> 29, 0);

    EXPECT_EQ(children[11] >> 29, 12);
    EXPECT_EQ(children[12] >> 29, 181);
    EXPECT_EQ(children[13] >> 29, 2729);
}

TEST(PancakeTest, OptPancake5GPU) {
    PancakeOptimizedGPU puz(5, false);
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
    EXPECT_EQ("0 1 2 4 3", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("0 1 4 3 2", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("0 4 3 2 1", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("4 3 2 1 0", puz.ToString(children[3]));
    EXPECT_EQ(3, ops[3]);
}

TEST(PancakeTest, OptPancake7GPU) {
    PancakeOptimizedGPU puz(7, false);
    std::string stateStr = "5 1 3 6 0 2 4";
    EXPECT_EQ(puz.ToString(puz.Parse(stateStr)), stateStr);
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(puz.Parse(stateStr));
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("5 1 3 6 0 4 2", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("5 1 3 6 4 2 0", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("5 1 3 4 2 0 6", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("5 1 4 2 0 6 3", puz.ToString(children[3]));
    EXPECT_EQ(3, ops[3]);
    EXPECT_EQ("5 4 2 0 6 3 1", puz.ToString(children[4]));
    EXPECT_EQ(4, ops[4]);
    EXPECT_EQ("4 2 0 6 3 1 5", puz.ToString(children[5]));
    EXPECT_EQ(5, ops[5]);
}

TEST(PancakeTest, OptPancakeGpu15) {
    PancakeOptimizedGPU puz(15, false);
    auto index = 0;
    auto stateStr = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14";
    EXPECT_EQ(puz.Parse(stateStr), index);
    EXPECT_EQ(puz.ToString(index), stateStr);
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 11 12 14 13", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 11 14 13 12", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 14 13 12 11", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 14 13 12 11 10", puz.ToString(children[3]));
    EXPECT_EQ(3, ops[3]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 14 13 12 11 10 9", puz.ToString(children[4]));
    EXPECT_EQ(4, ops[4]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 14 13 12 11 10 9 8", puz.ToString(children[5]));
    EXPECT_EQ(5, ops[5]);
    EXPECT_EQ("0 1 2 3 4 5 6 14 13 12 11 10 9 8 7", puz.ToString(children[6]));
    EXPECT_EQ(6, ops[6]);
    EXPECT_EQ("0 1 2 3 4 5 14 13 12 11 10 9 8 7 6", puz.ToString(children[7]));
    EXPECT_EQ(7, ops[7]);
    EXPECT_EQ("0 1 2 3 4 14 13 12 11 10 9 8 7 6 5", puz.ToString(children[8]));
    EXPECT_EQ(8, ops[8]);
    EXPECT_EQ("0 1 2 3 14 13 12 11 10 9 8 7 6 5 4", puz.ToString(children[9]));
    EXPECT_EQ(9, ops[9]);
    EXPECT_EQ("0 1 2 14 13 12 11 10 9 8 7 6 5 4 3", puz.ToString(children[10]));
    EXPECT_EQ(10, ops[10]);

    EXPECT_EQ("0 1 14 13 12 11 10 9 8 7 6 5 4 3 2", puz.ToString(children[11]));
    EXPECT_EQ(11, ops[11]);
    EXPECT_EQ("0 14 13 12 11 10 9 8 7 6 5 4 3 2 1", puz.ToString(children[12]));
    EXPECT_EQ(12, ops[12]);
    EXPECT_EQ("14 13 12 11 10 9 8 7 6 5 4 3 2 1 0", puz.ToString(children[13]));
    EXPECT_EQ(13, ops[13]);

    EXPECT_EQ(children[0] >> 29, 0);
    EXPECT_EQ(children[1] >> 29, 0);
    EXPECT_EQ(children[2] >> 29, 0);
    EXPECT_EQ(children[3] >> 29, 0);
    EXPECT_EQ(children[4] >> 29, 0);
    EXPECT_EQ(children[5] >> 29, 0);
    EXPECT_EQ(children[6] >> 29, 0);
    EXPECT_EQ(children[7] >> 29, 0);
    EXPECT_EQ(children[8] >> 29, 0);
    EXPECT_EQ(children[9] >> 29, 0);
    EXPECT_EQ(children[10] >> 29, 0);

    EXPECT_EQ(children[11] >> 29, 12);
    EXPECT_EQ(children[12] >> 29, 181);
    EXPECT_EQ(children[13] >> 29, 2729);
}

TEST(PancakeTest, OptPancakeGpu15Inv) {
    PancakeOptimizedGPU puz(15, true);
    auto index = 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 - 1;
    auto stateStr = "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14";
    EXPECT_EQ(puz.Parse(stateStr), index);
    EXPECT_EQ(puz.ToString(index), stateStr);
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 11 12 14 13", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 11 14 13 12", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 10 14 13 12 11", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 9 14 13 12 11 10", puz.ToString(children[3]));
    EXPECT_EQ(3, ops[3]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 8 14 13 12 11 10 9", puz.ToString(children[4]));
    EXPECT_EQ(4, ops[4]);
    EXPECT_EQ("0 1 2 3 4 5 6 7 14 13 12 11 10 9 8", puz.ToString(children[5]));
    EXPECT_EQ(5, ops[5]);
    EXPECT_EQ("0 1 2 3 4 5 6 14 13 12 11 10 9 8 7", puz.ToString(children[6]));
    EXPECT_EQ(6, ops[6]);
    EXPECT_EQ("0 1 2 3 4 5 14 13 12 11 10 9 8 7 6", puz.ToString(children[7]));
    EXPECT_EQ(7, ops[7]);
    EXPECT_EQ("0 1 2 3 4 14 13 12 11 10 9 8 7 6 5", puz.ToString(children[8]));
    EXPECT_EQ(8, ops[8]);
    EXPECT_EQ("0 1 2 3 14 13 12 11 10 9 8 7 6 5 4", puz.ToString(children[9]));
    EXPECT_EQ(9, ops[9]);
    EXPECT_EQ("0 1 2 14 13 12 11 10 9 8 7 6 5 4 3", puz.ToString(children[10]));
    EXPECT_EQ(10, ops[10]);

    EXPECT_EQ("0 1 14 13 12 11 10 9 8 7 6 5 4 3 2", puz.ToString(children[11]));
    EXPECT_EQ(11, ops[11]);
    EXPECT_EQ("0 14 13 12 11 10 9 8 7 6 5 4 3 2 1", puz.ToString(children[12]));
    EXPECT_EQ(12, ops[12]);
    EXPECT_EQ("14 13 12 11 10 9 8 7 6 5 4 3 2 1 0", puz.ToString(children[13]));
    EXPECT_EQ(13, ops[13]);

    EXPECT_EQ(children[0] >> 29, 0);
    EXPECT_EQ(children[1] >> 29, 0);
    EXPECT_EQ(children[2] >> 29, 0);
    EXPECT_EQ(children[3] >> 29, 0);
    EXPECT_EQ(children[4] >> 29, 0);
    EXPECT_EQ(children[5] >> 29, 0);
    EXPECT_EQ(children[6] >> 29, 0);
    EXPECT_EQ(children[7] >> 29, 0);
    EXPECT_EQ(children[8] >> 29, 0);
    EXPECT_EQ(children[9] >> 29, 0);
    EXPECT_EQ(children[10] >> 29, 0);

    EXPECT_EQ(children[11] >> 29, 12);
    EXPECT_EQ(children[12] >> 29, 181);
    EXPECT_EQ(children[13] >> 29, 2729);
}

