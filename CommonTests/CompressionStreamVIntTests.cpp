#include "pch.h"

#include "../Common/StreamVIntCompression.h"
#include "../Common/Util.h"

#include <vector>

static void TestStreamVInt(std::initializer_list<uint32_t> input, int expectedCount) {
	std::vector<uint32_t> indexes;
	for (auto i : input) indexes.push_back(i);
	std::vector<uint8_t> buffer(17 * indexes.size());
	auto encoded = StreamVIntCompression::Encode(indexes.size(), &indexes[0], &buffer[0]);
	EXPECT_EQ(encoded, expectedCount);
	std::vector<uint32_t> newindexes(indexes.size());
	auto decoded = StreamVIntCompression::Decode(indexes.size(), &buffer[0], &newindexes[0]);
	EXPECT_EQ(decoded, expectedCount);
	for (int i = 0; i < indexes.size(); i++) {
		EXPECT_EQ(newindexes[i], indexes[i]);
	}
	auto decoded2 = StreamVIntCompression::DecodeWithDiff(indexes.size(), &buffer[0], &newindexes[0]);
	EXPECT_EQ(decoded2, expectedCount);
	EXPECT_EQ(newindexes[0], indexes[0]);
	for (int i = 1; i < indexes.size(); i++) {
		EXPECT_EQ(newindexes[i] - newindexes[i - 1], indexes[i]);
	}
}

TEST(TestCompressionStreamVInt, StreamVInt) {
	TestStreamVInt({ 0, 1, 2, 3 }, 5);
	TestStreamVInt({ 256, 257, 258, 259 }, 9);
	TestStreamVInt({ 0x010000, 0x010001, 0x010002, 0x010003 }, 13);
	TestStreamVInt({ 0x01000000, 0x01000001, 0x01000002, 0x01000003 }, 17);

	TestStreamVInt({ 0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF }, 11);
	TestStreamVInt({ 0xFF00, 0xDEFF01, 0x01020304, 0x5 }, 11);
	TestStreamVInt({ 0x1E0F01, 0xFE020304, 0x5, 0x0505 }, 11);
	TestStreamVInt({ 0x01000000, 0x00, 0x0100, 0x010000 }, 11);

	TestStreamVInt({ 0, 0xFF00FF00, 1, 0xEEFFEEFF }, 11);

	TestStreamVInt({ 0, 0xFF44, 2, 3 }, 6);
	TestStreamVInt({ 0, 1, 0x01DE65, 3 }, 7);
	TestStreamVInt({ 0, 1, 2, 0x03344556 }, 8);

	TestStreamVInt({
			0, 1, 2, 3, 256, 257, 258, 259,
			0x010000, 0x010001, 0x010002, 0x010003,
			0x01000000, 0x01000001, 0x01000002, 0x01000003,
			0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF,
			0xFF00, 0xDEFF01, 0x01020304, 0x5,
			0x1E0F01, 0xFE020304, 0x5, 0x0505,
			0x01000000, 0x00, 0x0100, 0x010000,
			0, 0xFF00FF00, 1, 0xEEFFEEFF,
			0, 0xFF44, 2, 3,
			0, 1, 0x01DE65, 3,
			0, 1, 2, 0x03344556
		}, 120);

}
