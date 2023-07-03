#include "pch.h"

#include "../Common/StreamVInt.h"
#include "../Common/FrontierCompression.h"
#include "../Common/Util.h"

#include <vector>

TEST(TestFrontierCompression, TestBitmapCompression) {
	std::vector<uint32_t> indexes = { 7, 8, 32, 39, 177, 999, 1000, 1001, 1002, 1005 };
	std::vector<uint8_t> buffer(4096);
	int encodedSize = 
		FrontierCompression::EncodeBitMap(indexes.size(), &indexes[0], &buffer[0], buffer.size());

	std::vector<uint32_t> newindexes(1024);
	int size = encodedSize;
	int valsCount = FrontierCompression::DecodeBitMap(size, &buffer[0], &newindexes[0], newindexes.size());
	EXPECT_EQ(valsCount, indexes.size());
	for (int i = 0; i < indexes.size(); i++) {
		EXPECT_EQ(newindexes[i], indexes[i]);
	}
}

TEST(TestFrontierCompression, TestCompression_StreamVInt) {
	std::vector<uint32_t> indexes;
	for (int i = 0; i < 150; i++) {
		indexes.push_back(i * 11);
	}
	std::vector<uint8_t> buffer(4096);
	int encodedSize =
		FrontierCompression::Encode(indexes.size(), &indexes[0], &buffer[0], buffer.size());
	EXPECT_EQ(FrontierCompression::IsBitMap(&buffer[0]), false);
	std::vector<uint32_t> newindexes(1024);
	int size = encodedSize;
	int valsCount = FrontierCompression::Decode(size, &buffer[0], &newindexes[0], newindexes.size());
	EXPECT_EQ(valsCount, indexes.size());
	for (int i = 0; i < indexes.size(); i++) {
		EXPECT_EQ(newindexes[i], i * 11);
	}
}

TEST(TestFrontierCompression, TestCompression_Map) {
	std::vector<uint32_t> indexes;
	for (int i = 0; i < 150; i++) {
		indexes.push_back(i * 7);
	}
	std::vector<uint8_t> buffer(4096);
	int encodedSize =
		FrontierCompression::Encode(indexes.size(), &indexes[0], &buffer[0], buffer.size());
	EXPECT_EQ(FrontierCompression::IsBitMap(&buffer[0]), true);
	std::vector<uint32_t> newindexes(1024);
	int size = encodedSize;
	int valsCount = FrontierCompression::Decode(size, &buffer[0], &newindexes[0], newindexes.size());
	EXPECT_EQ(valsCount, indexes.size());
	for (int i = 0; i < indexes.size(); i++) {
		EXPECT_EQ(newindexes[i], indexes[i]);
	}
}

TEST(TestFrontierCompression, TestCompression_WithCheck) {
	std::vector<uint32_t> indexes;
	for (int i = 0; i < 150; i++) {
		indexes.push_back(i * 7);
	}
	indexes.push_back(700);
	auto orig_indexes = indexes;
	std::vector<uint8_t> buffer(4096);
	int encodedSize =
		FrontierCompression::EncodeWithCheck(indexes.size(), &indexes[0], &buffer[0], buffer.size());
	EXPECT_EQ(FrontierCompression::IsBitMap(&buffer[0]), false);
	std::vector<uint32_t> newindexes(1024);
	int size = encodedSize;
	int valsCount = FrontierCompression::Decode(size, &buffer[0], &newindexes[0], newindexes.size());
	EXPECT_EQ(valsCount, indexes.size());
	for (int i = 0; i < indexes.size(); i++) {
		EXPECT_EQ(newindexes[i], orig_indexes[i]);
	}
}
