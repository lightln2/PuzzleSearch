#include "pch.h"

#include "../ClassicBFS/PermutationMap.h"
#include "../SlidingTilePuzzle/Util.h"

#include <string>
#include <sstream>
#include <vector>

void TestCompact(std::string permutation, std::string expected) {
    std::vector<int> arr;

    {
        std::istringstream stream(permutation);
        while (!stream.eof()) {
            int val;
            stream >> val;
            arr.push_back(val);
        }
    }

    PermutationCompact(&arr[0], arr.size());

    {
        std::ostringstream output;
        for (int i = 0; i < arr.size(); i++) {
            if (i > 0) output << ' ';
            output << arr[i];
        }
        EXPECT_EQ(expected, output.str());
    }

    PermutationUncompact(&arr[0], arr.size());

    {
        std::ostringstream output;
        for (int i = 0; i < arr.size(); i++) {
            if (i > 0) output << ' ';
            output << arr[i];
        }
        EXPECT_EQ(permutation, output.str());
    }

}

void TestRank(std::string permutation, uint64_t expected) {
    std::vector<int> arr;

    {
        std::istringstream stream(permutation);
        while (!stream.eof()) {
            int val;
            stream >> val;
            arr.push_back(val);
        }
    }

    auto index = PermutationRank(&arr[0], arr.size());
    EXPECT_EQ(index, expected);

    PermutationUnrank(index, &arr[0], arr.size());

    {
        std::ostringstream output;
        for (int i = 0; i < arr.size(); i++) {
            if (i > 0) output << ' ';
            output << arr[i];
        }
        EXPECT_EQ(output.str(), permutation);
    }

}

TEST(TestPermutationMap, TestCompact) {
    TestCompact("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
    TestCompact("15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0", "15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0");
    TestCompact("8 9 10 11 0 1 2 3 4 5 6 7", "8 8 8 8 0 0 0 0 0 0 0 0");
    TestCompact("5 1 2 4 0 3", "5 1 1 2 0 0");
}

TEST(TestPermutationMap, TestRank) {
    TestRank("0 1 2", 0);
    TestRank("0 2 1", 1);
    TestRank("1 0 2", 2);
    TestRank("1 2 0", 3);
    TestRank("2 0 1", 4);
    TestRank("2 1 0", 5);

    TestRank("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15", 0);
    TestRank("15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0", 16ui64*15*14*13*12*11*10*9*8*7*6*5*4*3*2 - 1);

    TestRank("8 9 10 11 0 1 2 3 4 5 6 7", 351590400);
    TestRank("5 1 2 4 0 3", 634);
}

