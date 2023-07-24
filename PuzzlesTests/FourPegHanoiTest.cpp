#include "pch.h"

#include "../Puzzles/FourPegHanoiSimple.h"
#include "../Puzzles/FourPegHanoiOptimized.h"
#include "../Puzzles/FourPegHanoiGPU.h"
#include "../Puzzles/FourPegHanoiOptimizedGPU.h"
#include "../Common/Util.h"

#include <string>
#include <sstream>
#include <vector>

TEST(HanoiPuzzleTest, Simple5) {
    FourPegHanoiSimple puz(5);
    auto index = puz.Parse("1 2 3 4 5 0 0 0");
    EXPECT_EQ(index, 0);
    std::cerr << "Index: " << index << std::endl;
    EXPECT_EQ("1 2 3 4 5 0 0 0", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("2 3 4 5 0 0 1 0", puz.ToString(children[1]));
    EXPECT_EQ(2, ops[1]);
    EXPECT_EQ("2 3 4 5 0 0 0 1", puz.ToString(children[2]));
    EXPECT_EQ(3, ops[2]);
}

TEST(HanoiPuzzleTest, Simple7) {
    FourPegHanoiSimple puz(7);
    auto index = puz.Parse("1 5 6 0 2 0 0 3 4 7");
    EXPECT_EQ(index, 12532);
    std::cerr << "Index: " << index << std::endl;
    EXPECT_EQ("1 5 6 0 2 0 0 3 4 7", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("5 6 0 1 2 0 0 3 4 7", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("5 6 0 2 0 1 0 3 4 7", puz.ToString(children[1]));
    EXPECT_EQ(2, ops[1]);
    EXPECT_EQ("5 6 0 2 0 0 1 3 4 7", puz.ToString(children[2]));
    EXPECT_EQ(3, ops[2]);
    EXPECT_EQ("1 5 6 0 0 2 0 3 4 7", puz.ToString(children[3]));
    EXPECT_EQ(2, ops[3]);
    EXPECT_EQ("1 5 6 0 0 0 2 3 4 7", puz.ToString(children[4]));
    EXPECT_EQ(3, ops[4]);
    EXPECT_EQ("1 5 6 0 2 0 3 0 4 7", puz.ToString(children[5]));
    EXPECT_EQ(2, ops[5]);
}

TEST(HanoiPuzzleTest, Simple5_Symmetry) {
    FourPegHanoiSimple puz(5, true);
    auto index = puz.Parse("1 2 3 4 5 0 0 0");
    EXPECT_EQ(index, 0);
    std::cerr << "Index: " << index << std::endl;
    EXPECT_EQ("1 2 3 4 5 0 0 0", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[2]));
    EXPECT_EQ(1, ops[2]);
}

TEST(HanoiPuzzleTest, Simple7_Symmetry) {
    FourPegHanoiSimple puz(7, true);
    auto index = puz.Parse("3 4 6 0 1 5 7 0 2 0");
    EXPECT_EQ("3 4 6 0 1 5 7 0 2 0", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("1 3 4 6 0 5 7 0 2 0", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("2 3 4 6 0 1 5 7 0 0", puz.ToString(children[1]));
    EXPECT_EQ(0, ops[1]);
    EXPECT_EQ("4 6 0 1 5 7 0 3 0 2", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("3 4 6 0 5 7 0 1 2 0", puz.ToString(children[3]));
    EXPECT_EQ(2, ops[3]);
    EXPECT_EQ("3 4 6 0 5 7 0 2 0 1", puz.ToString(children[4]));
    EXPECT_EQ(3, ops[4]);
    EXPECT_EQ("3 4 6 0 1 5 7 0 2 0", puz.ToString(children[5]));
    EXPECT_EQ(2, ops[5]);
}

/////////// OPTIMIZED ///////////////////

TEST(HanoiPuzzleTest, Opt5) {
    FourPegHanoiOptimized puz(5);
    auto index = puz.Parse("1 2 3 4 5 0 0 0");
    EXPECT_EQ(index, 0);
    std::cerr << "Index: " << index << std::endl;
    EXPECT_EQ("1 2 3 4 5 0 0 0", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("2 3 4 5 0 0 1 0", puz.ToString(children[1]));
    EXPECT_EQ(2, ops[1]);
    EXPECT_EQ("2 3 4 5 0 0 0 1", puz.ToString(children[2]));
    EXPECT_EQ(3, ops[2]);
}

TEST(HanoiPuzzleTest, Opt7) {
    FourPegHanoiOptimized puz(7);
    auto index = puz.Parse("1 5 6 0 2 0 0 3 4 7");
    EXPECT_EQ(index, 12532);
    std::cerr << "Index: " << index << std::endl;
    EXPECT_EQ("1 5 6 0 2 0 0 3 4 7", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("5 6 0 1 2 0 0 3 4 7", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("5 6 0 2 0 1 0 3 4 7", puz.ToString(children[1]));
    EXPECT_EQ(2, ops[1]);
    EXPECT_EQ("5 6 0 2 0 0 1 3 4 7", puz.ToString(children[2]));
    EXPECT_EQ(3, ops[2]);
    EXPECT_EQ("1 5 6 0 0 2 0 3 4 7", puz.ToString(children[3]));
    EXPECT_EQ(2, ops[3]);
    EXPECT_EQ("1 5 6 0 0 0 2 3 4 7", puz.ToString(children[4]));
    EXPECT_EQ(3, ops[4]);
    EXPECT_EQ("1 5 6 0 2 0 3 0 4 7", puz.ToString(children[5]));
    EXPECT_EQ(2, ops[5]);
}

TEST(HanoiPuzzleTest, Opt5_Symmetry) {
    FourPegHanoiOptimized puz(5, true);
    auto index = puz.Parse("1 2 3 4 5 0 0 0");
    EXPECT_EQ(index, 0);
    std::cerr << "Index: " << index << std::endl;
    EXPECT_EQ("1 2 3 4 5 0 0 0", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[2]));
    EXPECT_EQ(1, ops[2]);
}

TEST(HanoiPuzzleTest, Opt7_Symmetry) {
    FourPegHanoiOptimized puz(7, true);
    auto index = puz.Parse("3 4 6 0 1 5 7 0 2 0");
    EXPECT_EQ("3 4 6 0 1 5 7 0 2 0", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("1 3 4 6 0 5 7 0 2 0", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("2 3 4 6 0 1 5 7 0 0", puz.ToString(children[1]));
    EXPECT_EQ(0, ops[1]);
    EXPECT_EQ("4 6 0 1 5 7 0 3 0 2", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("3 4 6 0 5 7 0 1 2 0", puz.ToString(children[3]));
    EXPECT_EQ(2, ops[3]);
    EXPECT_EQ("3 4 6 0 5 7 0 2 0 1", puz.ToString(children[4]));
    EXPECT_EQ(3, ops[4]);
    EXPECT_EQ("3 4 6 0 1 5 7 0 2 0", puz.ToString(children[5]));
    EXPECT_EQ(2, ops[5]);
}

TEST(HanoiPuzzleTest, Opt19_Symmetry) {
    FourPegHanoiOptimized puz(19, true);
    auto index = puz.Parse("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 0 0 0");
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 0 1 0 0", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 0 1 0 0", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
}

TEST(HanoiPuzzleTest, Opt19_Symmetry2) {
    FourPegHanoiOptimized puz(19, true);
    auto index = puz.Parse("18 19 0 13 14 15 16 17 0 1 2 3 4 5 6 7 8 9 10 11 12 0");
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("13 18 19 0 14 15 16 17 0 1 2 3 4 5 6 7 8 9 10 11 12 0", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("1 18 19 0 13 14 15 16 17 0 2 3 4 5 6 7 8 9 10 11 12 0", puz.ToString(children[1]));
    EXPECT_EQ(0, ops[1]);
    EXPECT_EQ("19 0 18 0 13 14 15 16 17 0 1 2 3 4 5 6 7 8 9 10 11 12", puz.ToString(children[2]));
    EXPECT_EQ(1, ops[2]);
    EXPECT_EQ("18 19 0 1 13 14 15 16 17 0 2 3 4 5 6 7 8 9 10 11 12 0", puz.ToString(children[3]));
    EXPECT_EQ(1, ops[3]);
    EXPECT_EQ("18 19 0 14 15 16 17 0 13 0 1 2 3 4 5 6 7 8 9 10 11 12", puz.ToString(children[4]));
    EXPECT_EQ(2, ops[4]);
    EXPECT_EQ("18 19 0 13 14 15 16 17 0 2 3 4 5 6 7 8 9 10 11 12 0 1", puz.ToString(children[5]));
    EXPECT_EQ(3, ops[5]);
}

/////////// GPU & OPTIMIZED ///////////////////

TEST(HanoiPuzzleTest, GpuOpt5) {
    FourPegHanoiOptimizedGPU puz(5, false);
    auto index = puz.Parse("1 2 3 4 5 0 0 0");
    EXPECT_EQ(index, 0);
    std::cerr << "Index: " << index << std::endl;
    EXPECT_EQ("1 2 3 4 5 0 0 0", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("2 3 4 5 0 0 1 0", puz.ToString(children[1]));
    EXPECT_EQ(2, ops[1]);
    EXPECT_EQ("2 3 4 5 0 0 0 1", puz.ToString(children[2]));
    EXPECT_EQ(3, ops[2]);
}

TEST(HanoiPuzzleTest, GpuOpt7) {
    FourPegHanoiOptimizedGPU puz(7, false);
    auto index = puz.Parse("1 5 6 0 2 0 0 3 4 7");
    EXPECT_EQ(index, 12532);
    std::cerr << "Index: " << index << std::endl;
    EXPECT_EQ("1 5 6 0 2 0 0 3 4 7", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("5 6 0 1 2 0 0 3 4 7", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("5 6 0 2 0 1 0 3 4 7", puz.ToString(children[1]));
    EXPECT_EQ(2, ops[1]);
    EXPECT_EQ("5 6 0 2 0 0 1 3 4 7", puz.ToString(children[2]));
    EXPECT_EQ(3, ops[2]);
    EXPECT_EQ("1 5 6 0 0 2 0 3 4 7", puz.ToString(children[3]));
    EXPECT_EQ(2, ops[3]);
    EXPECT_EQ("1 5 6 0 0 0 2 3 4 7", puz.ToString(children[4]));
    EXPECT_EQ(3, ops[4]);
    EXPECT_EQ("1 5 6 0 2 0 3 0 4 7", puz.ToString(children[5]));
    EXPECT_EQ(2, ops[5]);
}

TEST(HanoiPuzzleTest, GpuOpt5_Symmetry) {
    FourPegHanoiOptimizedGPU puz(5, true);
    auto index = puz.Parse("1 2 3 4 5 0 0 0");
    EXPECT_EQ(index, 0);
    std::cerr << "Index: " << index << std::endl;
    EXPECT_EQ("1 2 3 4 5 0 0 0", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
    EXPECT_EQ("2 3 4 5 0 1 0 0", puz.ToString(children[2]));
    EXPECT_EQ(1, ops[2]);
}

TEST(HanoiPuzzleTest, GpuOpt7_Symmetry) {
    FourPegHanoiOptimizedGPU puz(7, true);
    auto index = puz.Parse("3 4 6 0 1 5 7 0 2 0");
    EXPECT_EQ("3 4 6 0 1 5 7 0 2 0", puz.ToString(index));
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("1 3 4 6 0 5 7 0 2 0", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("2 3 4 6 0 1 5 7 0 0", puz.ToString(children[1]));
    EXPECT_EQ(0, ops[1]);
    EXPECT_EQ("4 6 0 1 5 7 0 3 0 2", puz.ToString(children[2]));
    EXPECT_EQ(2, ops[2]);
    EXPECT_EQ("3 4 6 0 5 7 0 1 2 0", puz.ToString(children[3]));
    EXPECT_EQ(2, ops[3]);
    EXPECT_EQ("3 4 6 0 5 7 0 2 0 1", puz.ToString(children[4]));
    EXPECT_EQ(3, ops[4]);
    EXPECT_EQ("3 4 6 0 1 5 7 0 2 0", puz.ToString(children[5]));
    EXPECT_EQ(2, ops[5]);
}

TEST(HanoiPuzzleTest, GpuOpt19_Symmetry) {
    FourPegHanoiOptimizedGPU puz(19, true);
    auto index = puz.Parse("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 0 0 0");
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 0 1 0 0", puz.ToString(children[0]));
    EXPECT_EQ(1, ops[0]);
    EXPECT_EQ("2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 0 1 0 0", puz.ToString(children[1]));
    EXPECT_EQ(1, ops[1]);
}

TEST(HanoiPuzzleTest, GpuOpt19_Symmetry2) {
    FourPegHanoiOptimizedGPU puz(19, true);
    auto index = puz.Parse("18 19 0 13 14 15 16 17 0 1 2 3 4 5 6 7 8 9 10 11 12 0");
    std::vector<uint64_t> indexes;
    std::vector<int> usedOpBits;
    std::vector<uint64_t> children;
    std::vector<int> ops;
    indexes.push_back(index);
    usedOpBits.push_back(0);
    puz.Expand(indexes, usedOpBits, children, ops, {});
    EXPECT_EQ("13 18 19 0 14 15 16 17 0 1 2 3 4 5 6 7 8 9 10 11 12 0", puz.ToString(children[0]));
    EXPECT_EQ(0, ops[0]);
    EXPECT_EQ("1 18 19 0 13 14 15 16 17 0 2 3 4 5 6 7 8 9 10 11 12 0", puz.ToString(children[1]));
    EXPECT_EQ(0, ops[1]);
    EXPECT_EQ("19 0 18 0 13 14 15 16 17 0 1 2 3 4 5 6 7 8 9 10 11 12", puz.ToString(children[2]));
    EXPECT_EQ(1, ops[2]);
    EXPECT_EQ("18 19 0 1 13 14 15 16 17 0 2 3 4 5 6 7 8 9 10 11 12 0", puz.ToString(children[3]));
    EXPECT_EQ(1, ops[3]);
    EXPECT_EQ("18 19 0 14 15 16 17 0 13 0 1 2 3 4 5 6 7 8 9 10 11 12", puz.ToString(children[4]));
    EXPECT_EQ(2, ops[4]);
    EXPECT_EQ("18 19 0 13 14 15 16 17 0 2 3 4 5 6 7 8 9 10 11 12 0 1", puz.ToString(children[5]));
    EXPECT_EQ(3, ops[5]);
}
