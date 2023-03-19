#include "pch.h"

#include "../SlidingTilePuzzle/StreamVInt.h"
#include "../SlidingTilePuzzle/Util.h"

#include <vector>

static void TestTuple(std::initializer_list<uint32_t> input, int expected) {
	uint32_t indexes[4];
	int pos = 0;
	for (auto i : input) {
		indexes[pos++] = i;
	}
	uint8_t buffer[17];
	int encoded = StreamVInt::EncodeTuple(indexes, buffer);
	EXPECT_EQ(encoded, expected);
	uint32_t newindexes[4];
	int decoded = StreamVInt::DecodeTuple(buffer, newindexes);
	EXPECT_EQ(decoded, expected);
	for (int i = 0; i < 4; i++) {
		EXPECT_EQ(newindexes[i], indexes[i]);
	}
}

TEST(TestStreamVInt, EncodeTuple) {
	TestTuple({ 0, 1, 2, 3 }, 5);
	TestTuple({ 256, 257, 258, 259 }, 9);
	TestTuple({ 0x010000, 0x010001, 0x010002, 0x010003 }, 13);
	TestTuple({ 0x01000000, 0x01000001, 0x01000002, 0x01000003 }, 17);

	TestTuple({ 0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF }, 11);
	TestTuple({ 0xFF00, 0xDEFF01, 0x01020304, 0x5 }, 11);
	TestTuple({ 0x1E0F01, 0xFE020304, 0x5, 0x0505 }, 11);
	TestTuple({ 0x01000000, 0x00, 0x0100, 0x010000 }, 11);

	TestTuple({ 0, 0xFF00FF00, 1, 0xEEFFEEFF }, 11);

	TestTuple({ 0, 0xFF44, 2, 3 }, 6);
	TestTuple({ 0, 1, 0x01DE65, 3 }, 7);
	TestTuple({ 0, 1, 2, 0x03344556 }, 8);
}

static void TestBoundsTuple(std::initializer_list<uint8_t> input) {
	uint8_t bounds[8];
	int pos = 0;
	for (auto i : input) {
		bounds[pos++] = i;
	}
	uint8_t buffer[4];
	StreamVInt::EncodeBoundsTuple(bounds, buffer);
	uint8_t newbounds[8];
	StreamVInt::DecodeBoundsTuple(buffer, newbounds);
	for (int i = 0; i < 8; i++) {
		EXPECT_EQ(newbounds[i], bounds[i]);
	}
}

TEST(TestStreamVInt, EncodeBoundsTuple) {
	TestBoundsTuple({ 0, 1, 2, 3, 4, 5, 6, 7 });
	TestBoundsTuple({ 8, 9, 10, 11, 12, 13, 14, 15 });
	TestBoundsTuple({ 0, 15, 2, 7, 5, 2, 3, 11 });
}

static void TestIndexesTuples(std::initializer_list<uint32_t> input, int expected) {
	std::vector<uint32_t> indexes;
	for (auto i : input) {
		indexes.push_back(i);
	}
	std::vector<uint8_t> buffer(17 * indexes.size());
	int encoded = StreamVInt::EncodeIndexes(indexes.size(), &indexes[0], &buffer[0]);
	EXPECT_EQ(encoded, expected);
	std::vector<uint32_t> newindexes(indexes.size());
	int decoded = StreamVInt::DecodeIndexes(indexes.size(), &buffer[0], &newindexes[0]);
	EXPECT_EQ(decoded, expected);
	for (int i = 0; i < indexes.size(); i++) {
		EXPECT_EQ(newindexes[i], indexes[i]);
	}
}

static void TestBoundsTuples(std::initializer_list<uint8_t> input) {
	std::vector<uint8_t> bounds;
	for (auto i : input) {
		bounds.push_back(i);
	}
	std::vector<uint8_t> buffer(bounds.size() / 2);
	int encoded = StreamVInt::EncodeBounds(bounds.size(), &bounds[0], &buffer[0]);
	EXPECT_EQ(encoded, bounds.size() / 2);
	std::vector<uint8_t> newbounds(bounds.size());
	int decoded = StreamVInt::DecodeBounds(bounds.size(), &buffer[0], &newbounds[0]);
	EXPECT_EQ(decoded, bounds.size() / 2);
	for (int i = 0; i < bounds.size(); i++) {
		EXPECT_EQ(newbounds[i], bounds[i]);
	}
}

TEST(TestStreamVInt, EncodeTuplesList) {
	TestIndexesTuples(
		{ 
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

	TestBoundsTuples(
		{
			0, 1, 2, 3, 4, 5, 6, 7,
			8, 9, 10, 11, 12, 13, 14, 15,
			0, 15, 2, 7, 5, 2, 3, 11
		});

}

TEST(TestStreamVInt, EncodeFrontier) {
	Buffer<uint32_t> indexes(StreamVInt::MAX_INDEXES_COUNT);
	Buffer<uint8_t> bounds(StreamVInt::MAX_INDEXES_COUNT);
	Buffer<uint8_t> buffer(StreamVInt::MAX_BUFFER_SIZE);
	for (int i = 0; i < 65536; i++) {
		indexes.Add(i * i);
		bounds.Add(i % 16);
	}
	StreamVInt::Encode(indexes, bounds, buffer);

	int position = 0;
	position = StreamVInt::Decode(0, buffer, indexes, bounds);
	for (int i = 0; i < 65536; i++) {
		ENSURE_EQ(indexes[i], i * i);
		ENSURE_EQ(bounds[i], i % 16);
	}
}

TEST(TestStreamVInt, EncodeExpanded) {
	Buffer<uint32_t> indexes(StreamVInt::MAX_INDEXES_COUNT);
	Buffer<uint8_t> buffer(StreamVInt::MAX_BUFFER_SIZE);
	for (int i = 0; i < 65536; i++) {
		indexes.Add(i * i);
	}
	StreamVInt::Encode(indexes, buffer);

	int position = 0;
	position = StreamVInt::Decode(0, buffer, indexes);
	for (int i = 0; i < 65536; i++) {
		ENSURE_EQ(indexes[i], i * i);
	}
}

TEST(TestStreamVInt, EncodeFrontierChunks) {
	Buffer<uint32_t> indexes(StreamVInt::MAX_INDEXES_COUNT);
	Buffer<uint8_t> bounds(StreamVInt::MAX_INDEXES_COUNT);
	Buffer<uint8_t> buffer(StreamVInt::MAX_BUFFER_SIZE * 10);
	for (int n = 0; n < 10; n++) {
		indexes.Clear();
		bounds.Clear();
		for (int i = 0; i < 65536; i++) {
			indexes.Add(i * i);
			bounds.Add(i % 16);
		}
		StreamVInt::Encode(indexes, bounds, buffer);
	}

	int position = 0;
	for (int n = 0; n < 10; n++) {
		position = StreamVInt::Decode(position, buffer, indexes, bounds);
		for (int i = 0; i < 65536; i++) {
			ENSURE_EQ(indexes[i], i * i);
			ENSURE_EQ(bounds[i], i % 16);
		}
	}
}

TEST(TestStreamVInt, EncodeExpandedChunks) {
	Buffer<uint32_t> indexes(StreamVInt::MAX_INDEXES_COUNT);
	Buffer<uint8_t> buffer(StreamVInt::MAX_BUFFER_SIZE * 10);
	for (int n = 0; n < 10; n++) {
		indexes.Clear();
		for (int i = 0; i < 65536; i++) {
			indexes.Add(i * i);
		}
		StreamVInt::Encode(indexes, buffer);
	}

	int position = 0;
	for (int n = 0; n < 10; n++) {
		position = StreamVInt::Decode(position, buffer, indexes);
		for (int i = 0; i < 65536; i++) {
			ENSURE_EQ(indexes[i], i * i);
		}
	}
}
