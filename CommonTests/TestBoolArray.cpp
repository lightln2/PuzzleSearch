#include "pch.h"

#include "../Common/BoolArray.h"
#include "../Common/Util.h"

TEST(BoolArrayTests, TestBoolArray) {
    constexpr int TRY = 2;
    constexpr uint64_t MAX = 1ui64 << 32;
    constexpr int INC = 13;
    BoolArray ba(MAX);
    uint64_t totalNanos = 0;
    for (int t = 0; t < TRY; t++) {
        Timer t1;
        for (uint64_t i = 0; i < MAX; i += INC) {
            ba.Set(i);
        }
        std::cerr << t1 << std::endl;
        Timer t2;
        uint64_t res = 0;
        ba.ScanBitsAndClear([&](uint64_t index) {
            res += index;
            });
        totalNanos += t2.Elapsed();
        ensure(res == 709490155855181350ui64);
        std::cerr << t2 << ": " << res << std::endl;
    }
    std::cerr << "Total: " << WithTime(totalNanos) << std::endl;

    {
        uint64_t res = 0;
        ba.ScanBitsAndClear([&](uint64_t index) {
            res += index;
        });
        ensure(res == 0);

    }
}
