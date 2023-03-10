#include "pch.h"

#include "../SlidingTilePuzzle/Puzzle.h"
#include "../SlidingTilePuzzle/Util.h"

TEST(TestPuzzle, CanMove) {
	uint32_t base = 16 * 1234;
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 0)), false);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 1)), false);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 2)), false);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 3)), false);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 4)), true);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 5)), true);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 6)), true);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 7)), true);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 8)), true);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 9)), true);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 10)), true);
	EXPECT_EQ((Puzzle<4, 3>::CanMoveUp(base + 11)), true);

	EXPECT_EQ((Puzzle<5, 3>::CanMoveDown(base + 13)), false);
	EXPECT_EQ((Puzzle<8, 2>::CanMoveLeft(base + 8)), false);
	EXPECT_EQ((Puzzle<8, 2>::CanMoveRight(base + 8)), true);

	EXPECT_EQ((Puzzle<8, 2>::GetBounds(base + 0)), 5);
}

template<int width, int height>
void TestHashing(std::string puzzleStr,
				 std::string unpackedStr,
				 std::string packedStr, 
				 uint32_t segment,
				 uint32_t index)
{
	Puzzle<width, height> puzzle;
	auto unpackedState = puzzle.Parse(puzzleStr);
	EXPECT_EQ(unpackedState.ToString(), unpackedStr);
	auto packedState = puzzle.Pack(unpackedState);
	EXPECT_EQ(packedState.ToString(), packedStr);
	auto stateHash = puzzle.GetIndex(packedState);
	EXPECT_EQ(stateHash.first, segment);
	EXPECT_EQ(stateHash.second, index);
	packedState = puzzle.FromIndex(stateHash.first, stateHash.second);
	EXPECT_EQ(packedState.ToString(), packedStr);
	unpackedState = puzzle.Unpack(packedState);
	EXPECT_EQ(unpackedState.ToString(), unpackedStr);
	EXPECT_EQ((puzzle.ToString(unpackedState)), puzzleStr);

	stateHash = puzzle.Rank(puzzleStr);
	EXPECT_EQ(stateHash.first, segment);
	EXPECT_EQ(stateHash.second, index);
	EXPECT_EQ((puzzle.Unrank(stateHash)), puzzleStr);
}

TEST(TestPuzzle, Hashing4x4) {
	TestHashing<4, 4>("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
					  "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 b=0",
					  "0 0 2 3 4 5 6 7 8 9 10 11 12 13 14 b=0",
					  Puzzle<4, 4>::MaxSegments() - 1, Puzzle<4, 4>::MaxIndexesPerSegment() - 16);
	TestHashing<4, 4>("15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0",
					  "14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 b=15",
					  "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 b=15",
					  0, 15);
	TestHashing<4, 4>("15 14 8 6 1 2 9 13 12 5 0 3 4 10 11 7",
					  "14 13 7 5 0 1 8 12 11 4 2 3 9 10 6 b=10",
					  "0 0 0 0 0 1 4 5 5 2 2 3 8 9 6 b=10",
					  0x698, 1023708490);
}

TEST(TestPuzzle, Hashing5x3) {
	TestHashing<5, 3>("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14",
					  "0 1 2 3 4 5 6 7 8 9 10 11 12 13 0 b=0",
					  "0 0 2 3 4 5 6 7 8 9 10 11 12 13 0 b=0",
					  Puzzle<5, 3>::MaxSegments() - 1, Puzzle<5, 3>::MaxIndexesPerSegment() - 16);
	TestHashing<5, 3>("13 14 12 11 10 9 8 7 6 5 4 3 2 1 0",
					  "12 13 11 10 9 8 7 6 5 4 3 2 1 0 0 b=14",
					  "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 b=14",
					  0, 14);
	TestHashing<5, 3>("14 8 6 1 2 9 13 12 5 0 3 4 10 11 7",
					  "13 7 5 0 1 8 12 11 4 2 3 9 10 6 0 b=9",
					  "0 0 0 0 1 4 5 5 2 2 3 8 9 6 0 b=9",
					  0x69, 2648452041);
}

TEST(TestPuzzle, Hashing7x2) {
	TestHashing<7, 2>("0 1 2 3 4 5 6 7 8 9 10 11 12 13",
					  "0 1 2 3 4 5 6 7 8 9 10 11 12 0 0 b=0",
					  "0 0 2 3 4 5 6 7 8 9 10 11 12 0 0 b=0",
					  Puzzle<7, 2>::MaxSegments() - 1, Puzzle<7, 2>::MaxIndexesPerSegment() - 16);
	TestHashing<7, 2>("13 12 11 10 9 8 7 6 5 4 3 2 1 0",
					  "12 11 10 9 8 7 6 5 4 3 2 1 0 0 0 b=13",
					  "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 b=13",
					  0, 13);
	TestHashing<7, 2>("6 8 1 2 9 13 12 5 0 3 4 10 11 7",
					  "5 7 0 1 8 12 11 4 2 3 9 10 6 0 0 b=8",
					  "0 0 0 1 4 5 5 2 2 3 8 9 6 0 0 b=8",
					  0x6, 3115722104);
}

TEST(TestPuzzle, Hashing4x3) {
	TestHashing<4, 3>("0 1 2 3 4 5 6 7 8 9 10 11",
					  "0 1 2 3 4 5 6 7 8 9 10 0 0 0 0 b=0",
					  "0 0 2 3 4 5 6 7 8 9 10 0 0 0 0 b=0",
					  Puzzle<4, 3>::MaxSegments() - 1, Puzzle<4, 3>::MaxIndexesPerSegment() - 16);
	TestHashing<4, 3>("10 11 9 8 7 6 5 4 3 2 1 0",
					  "9 10 8 7 6 5 4 3 2 1 0 0 0 0 0 b=11",
					  "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 b=11",
					  0, 11);
	TestHashing<4, 3>("6 8 1 2 9 5 0 3 4 10 11 7",
					  "5 7 0 1 8 4 2 3 9 10 6 0 0 0 0 b=6",
					  "0 0 0 1 4 2 2 3 8 9 6 0 0 0 0 b=6",
					  0, 203025462);
}

template<int width, int height>
void TestMoveUp(std::string from, std::string to) {
	Puzzle<width, height> puzzle;
	auto stateHash = puzzle.Rank(from);
	EXPECT_EQ((puzzle.CanMoveUp(stateHash.second)), true);
	stateHash = puzzle.MoveUp(stateHash);
	EXPECT_EQ((puzzle.Unrank(stateHash)), to);
}

template<int width, int height>
void TestMoveDown(std::string from, std::string to) {
	Puzzle<width, height> puzzle;
	auto stateHash = puzzle.Rank(from);
	EXPECT_EQ((puzzle.CanMoveDown(stateHash.second)), true);
	stateHash = puzzle.MoveDown(stateHash);
	EXPECT_EQ((puzzle.Unrank(stateHash)), to);
}

template<int width, int height>
void TestMoveLeft(std::string from, std::string to) {
	Puzzle<width, height> puzzle;
	auto stateHash = puzzle.Rank(from);
	EXPECT_EQ((puzzle.CanMoveLeft(stateHash.second)), true);
	stateHash.second = puzzle.MoveLeft(stateHash.second);
	EXPECT_EQ((puzzle.Unrank(stateHash)), to);
}

template<int width, int height>
void TestMoveRight(std::string from, std::string to) {
	Puzzle<width, height> puzzle;
	auto stateHash = puzzle.Rank(from);
	EXPECT_EQ((puzzle.CanMoveRight(stateHash.second)), true);
	stateHash.second = puzzle.MoveRight(stateHash.second);
	EXPECT_EQ((puzzle.Unrank(stateHash)), to);
}

TEST(TestPuzzle, Move4x4) {
	TestMoveDown<4, 4>("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
					   "4 1 2 3 0 5 6 7 8 9 10 11 12 13 14 15");
	TestMoveDown<4, 4>("4 1 2 3 0 5 6 7 8 9 10 11 12 13 14 15",
					   "4 1 2 3 8 5 6 7 0 9 10 11 12 13 14 15");
	TestMoveDown<4, 4>("4 1 2 3 8 5 6 7 0 9 10 11 12 13 14 15",
					   "4 1 2 3 8 5 6 7 12 9 10 11 0 13 14 15");

	TestMoveUp<4, 4>("15 14 8 6 1 2 9 13 12 5 0 3 4 10 11 7",
					 "15 14 8 6 1 2 0 13 12 5 9 3 4 10 11 7");
	TestMoveDown<4, 4>("15 14 8 6 1 2 9 13 12 5 0 3 4 10 11 7",
					   "15 14 8 6 1 2 9 13 12 5 11 3 4 10 0 7");
	TestMoveLeft<4, 4>("15 14 8 6 1 2 9 13 12 5 0 3 4 10 11 7",
					   "15 14 8 6 1 2 9 13 12 0 5 3 4 10 11 7");
	TestMoveRight<4, 4>("15 14 8 6 1 2 9 13 12 5 0 3 4 10 11 7",
					    "15 14 8 6 1 2 9 13 12 5 3 0 4 10 11 7");
}

TEST(TestPuzzle, Move8x2) {
	TestMoveDown<8, 2>("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
					   "8 1 2 3 4 5 6 7 0 9 10 11 12 13 14 15");

	TestMoveUp<8, 2>("14 15 8 6 1 2 9 13 12 5 0 3 4 10 11 7",
					 "14 15 0 6 1 2 9 13 12 5 8 3 4 10 11 7");
	TestMoveLeft<8, 2>("14 15 8 6 1 2 9 13 12 5 0 3 4 10 11 7",
					   "14 15 8 6 1 2 9 13 12 0 5 3 4 10 11 7");
	TestMoveRight<8, 2>("14 15 8 6 1 2 9 13 12 5 0 3 4 10 11 7",
					    "14 15 8 6 1 2 9 13 12 5 3 0 4 10 11 7");

	TestMoveDown<8, 2>("14 0 8 6 1 2 9 13 12 5 15 3 4 10 7 11",
					   "14 5 8 6 1 2 9 13 12 0 15 3 4 10 7 11");
	TestMoveUp<8, 2>("14 5 8 6 1 2 9 13 12 0 15 3 4 10 7 11",
				     "14 0 8 6 1 2 9 13 12 5 15 3 4 10 7 11");
}

TEST(TestPuzzle, Move5x3) {
	TestMoveDown<5, 3>("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14",
					   "5 1 2 3 4 0 6 7 8 9 10 11 12 13 14");
	TestMoveRight<5, 3>("5 1 2 3 4 0 6 7 8 9 10 11 12 13 14",
					    "5 1 2 3 4 6 0 7 8 9 10 11 12 13 14");
	TestMoveUp<5, 3>("5 1 2 3 4 6 0 7 8 9 10 11 12 13 14",
					 "5 0 2 3 4 6 1 7 8 9 10 11 12 13 14");
	TestMoveLeft<5, 3>("5 0 2 3 4 6 1 7 8 9 10 11 12 13 14",
					   "0 5 2 3 4 6 1 7 8 9 10 11 12 13 14");
}

TEST(TestPuzzle, Move2x2) {
	TestMoveDown<2, 2>("0 1 2 3", "2 1 0 3");
	TestMoveRight<2, 2>("2 1 0 3", "2 1 3 0");
	TestMoveUp<2, 2>("2 1 3 0", "2 0 3 1");
	TestMoveLeft<2, 2>("2 0 3 1", "0 2 3 1");
}

TEST(TestPuzzle , TestPerformance4x4) {
	Puzzle<4, 4> puzzle;
	constexpr int MAX = 1 * 1000 * 1000;
	uint64_t hashUp = 0, hashDown = 0;

	int SEGMENT = 0x5a2;

	for (uint32_t i = 0; i < MAX; i++) {
		int blank = i % 16;
		if (puzzle.CanMoveUp(i)) {
			auto res = puzzle.MoveUp(SEGMENT, i);
			hashUp = hashUp * 31 + (uint64_t(res.first) << 32) + res.second;
			ENSURE_EQ(SEGMENT != res.first, puzzle.UpChangesSegment(blank));
		}
		if (puzzle.CanMoveDown(i)) {
			auto res = puzzle.MoveDown(SEGMENT, i);
			hashDown = hashDown * 31 + (uint64_t(res.first) << 32) + res.second;
			ENSURE_EQ(SEGMENT != res.first, puzzle.DownChangesSegment(blank));
		}
	}

	if (MAX == 10 * 1000 * 1000) {
		EXPECT_EQ(hashUp, 15956298610708895712);
		EXPECT_EQ(hashDown, 8254655271410123088);
	}
	else if (MAX == 1 * 1000 * 1000) {
		EXPECT_EQ(hashUp, 15495615531757136264);
		EXPECT_EQ(hashDown, 3846743369661599048);
	}
}
