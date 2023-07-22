#include "pch.h"

#include "../Common/OpBitsCompression.h"
#include "../Common/Util.h"

#include <vector>

template<int BITS>
static void TestOpBits(std::initializer_list<uint8_t> input, size_t expectedCount) {
	std::vector<uint8_t> values;
	for (auto i : input) values.push_back(i);
	std::vector<uint8_t> buffer(values.size());
	auto encoded = OpBitsCompression<BITS>::Encode(values.size(), &values[0], &buffer[0]);
	EXPECT_EQ(encoded, expectedCount);
	std::vector<uint8_t> newvalues(values.size());
	auto decoded = OpBitsCompression<BITS>::Decode(values.size(), &buffer[0], &newvalues[0]);
	EXPECT_EQ(decoded, expectedCount);
	for (int i = 0; i < values.size(); i++) {
		EXPECT_EQ(newvalues[i], values[i]);
	}
}

TEST(TestCompressionOpBits, OpBits2) {
	TestOpBits<2>({ 0, 1, 2, 3, 2, 1, 0, 3 }, 2);
	TestOpBits<2>({ 3, 2, 3, 1, 1, 0, 0, 2 }, 2);
	TestOpBits<2>({ 0, 1, 2, 3, 2, 1, 0, 3, 3, 2, 3, 1, 1, 0, 0, 2 }, 4);
}

TEST(TestCompressionOpBits, OpBits4) {
	TestOpBits<4>({ 0, 1, 2, 3, 15, 14, 13, 12 }, 4);
	TestOpBits<4>({ 13, 4, 12, 15, 15, 0, 0, 7 }, 4);
	TestOpBits<4>({ 0, 1, 2, 3, 15, 14, 13, 12, 13, 4, 12, 15, 15, 0, 0, 7 }, 8);
}

TEST(TestCompressionOpBits, OpBits8) {
	TestOpBits<8>({ 0, 1, 2, 13, 25, 74, 143, 241 }, 8);
	TestOpBits<8>({ 255, 254, 127, 128, 19, 0, 0, 248 }, 8);
	TestOpBits<8>({ 0, 1, 2, 13, 25, 74, 143, 241, 255, 254, 127, 128, 19, 0, 0, 248 }, 16);
}
