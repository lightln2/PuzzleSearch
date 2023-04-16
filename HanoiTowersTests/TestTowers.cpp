#include "pch.h"

#include "../HanoiTowers/Towers.h"

TEST(TestTowers, TestRankUnrank) {
    HanoiTowers<18> towers;
    auto initialState = towers.InitialState;
    auto initialIndex = initialState >> 2;
    EXPECT_EQ(towers.ToString(initialState), 
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 last=1");
    UnpackedHT unpacked;
    unpacked.Unpack(18, initialIndex);
    EXPECT_EQ(unpacked.ToString(), "0: [ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 ] 1: [ ] 2: [ ] 3: [ ]");

}
