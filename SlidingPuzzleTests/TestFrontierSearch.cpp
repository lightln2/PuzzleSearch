#include "pch.h"

#include "../SlidingTilePuzzle/FrontierSearch.h"
#include "../SlidingTilePuzzle/Puzzle.h"

template<int width, int height>
void TestSearch(size_t expected_radius, std::string counts, SearchOptions options = {}) {
	auto result = FrontierSearch<width, height>(options);
	EXPECT_EQ(result.size() - 1, expected_radius);

	std::ostringstream stream;
	for (size_t i = 0; i < result.size(); i++) {
		if (i > 0) stream << ' ';
		stream << result[i];
	}
	EXPECT_EQ((stream.str()), counts);
}

TEST(TestFrontierSearch, Search2x2) {
	TestSearch<2, 2>(6, "1 2 2 2 2 2 1");
}

TEST(TestFrontierSearch, Search3x2) {
	TestSearch<3, 2>(21, "1 2 3 5 6 7 10 12 12 16 23 25 28 39 44 40 29 21 18 12 6 1");
}

TEST(TestFrontierSearch, Search4x2) {
	TestSearch<4, 2>(36, "1 2 3 6 10 14 19 28 42 61 85 119 161 215 293 396 506 632 788 985 1194 1414 1664 1884 1999 1958 1770 1463 1076 667 361 190 88 39 19 7 1");
}

TEST(TestFrontierSearch, Search3x3) {
	TestSearch<3, 3>(31, "1 2 4 8 16 20 39 62 116 152 286 396 748 1024 1893 2512 4485 5638 9529 10878 16993 17110 23952 20224 24047 15578 14560 6274 3910 760 221 2");
}

TEST(TestFrontierSearch, Search5x2) {
	TestSearch<5, 2>(55, "1 2 3 6 11 19 30 44 68 112 176 271 411 602 851 1232 1783 2530 3567 4996 6838 9279 12463 16597 21848 28227 35682 44464 54597 65966 78433 91725 104896 116966 126335 131998 133107 128720 119332 106335 91545 75742 60119 45840 33422 23223 15140 9094 5073 2605 1224 528 225 75 20 2");
}

TEST(TestFrontierSearch, Search6x2) {
	SearchOptions opts;
	opts.MaxDepth = 10;
	TestSearch<6, 2>(10, "1 2 3 6 11 20 36 60 95 155 258", opts);
}

TEST(TestFrontierSearch, Search4x3) {
	SearchOptions opts;
	opts.MaxDepth = 10;
	TestSearch<4, 3>(10, "1 2 4 9 20 37 63 122 232 431 781", opts);
}

TEST(TestFrontierSearch, Search7x2) {
	SearchOptions opts;
	opts.MaxDepth = 10;
	TestSearch<7, 2>(10, "1 2 3 6 11 20 37 67 117 198 329", opts);
}

TEST(TestFrontierSearch, Search5x3) {
	SearchOptions opts;
	//opts.MaxDepth = 8;
	//TestSearch<5, 3>(8, "1 2 4 9 21 42 89 164 349", opts);
	opts.MaxDepth = 7;
	TestSearch<5, 3>(7, "1 2 4 9 21 42 89 164", opts);
}

TEST(TestFrontierSearch, Search8x2) {
	SearchOptions opts;
	opts.MaxDepth = 5;
	TestSearch<8, 2>(5, "1 2 3 6 11 20", opts);
}

TEST(TestFrontierSearch, Search4x4) {
	SearchOptions opts;
	opts.MaxDepth = 5;
	TestSearch<4, 4>(5, "1 2 4 10 24 54", opts);
}
