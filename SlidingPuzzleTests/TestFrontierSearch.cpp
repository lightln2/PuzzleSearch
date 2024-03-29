#include "pch.h"

#include "../SlidingTilePuzzle/FrontierSearch.h"
#include "../SlidingTilePuzzle/Puzzle.h"

template<int width, int height>
void TestSearch(size_t expected_radius, std::string counts, STSearchOptions options = {}) {
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

TEST(TestFrontierSearch, Search3x2_edge) {
	STSearchOptions opts;
	opts.InitialValue = "1 0 2  3 4 5";
	TestSearch<3, 2>(21, "1 3 4 4 6 10 10 10 16 20 20 26 36 40 40 37 29 20 14 9 4 1", opts);
}

TEST(TestFrontierSearch, Search4x2) {
	TestSearch<4, 2>(36, "1 2 3 6 10 14 19 28 42 61 85 119 161 215 293 396 506 632 788 985 1194 1414 1664 1884 1999 1958 1770 1463 1076 667 361 190 88 39 19 7 1");
}

TEST(TestFrontierSearch, Search4x2_edge) {
	STSearchOptions opts;
	opts.InitialValue = "1 0 2 3  4 5 6 7";
	TestSearch<4, 2>(35,
		"1 3 5 7 10 16 24 34 49 72 100 134 182 252 339 439 557 714 892 1082 1281 1503 1741 1913 1963 1883 1681 1330 887 512 280 146 72 36 16 4", opts);
}

TEST(TestFrontierSearch, Search3x3) {
	TestSearch<3, 3>(31, "1 2 4 8 16 20 39 62 116 152 286 396 748 1024 1893 2512 4485 5638 9529 10878 16993 17110 23952 20224 24047 15578 14560 6274 3910 760 221 2");
}

TEST(TestFrontierSearch, Search3x3_edge) {
	STSearchOptions opts;
	opts.InitialValue = "1 0 2  3 4 5  6 7 8";
	TestSearch<3, 3>(31,
		"1 3 5 10 14 28 42 80 108 202 278 524 726 1348 1804 3283 4193 7322 8596 13930 14713 21721 19827 25132 18197 18978 9929 7359 2081 878 126 2", opts);
}

TEST(TestFrontierSearch, Search3x3_center) {
	STSearchOptions opts;
	opts.InitialValue = "1 2 3  4 0 5  6 7 8";
	TestSearch<3, 3>(30,
		"1 4 8 8 16 32 60 72 136 200 376 512 964 1296 2368 3084 5482 6736 11132 12208 18612 18444 24968 19632 22289 13600 11842 4340 2398 472 148", opts);
}

TEST(TestFrontierSearch, Search5x2) {
	TestSearch<5, 2>(55, "1 2 3 6 11 19 30 44 68 112 176 271 411 602 851 1232 1783 2530 3567 4996 6838 9279 12463 16597 21848 28227 35682 44464 54597 65966 78433 91725 104896 116966 126335 131998 133107 128720 119332 106335 91545 75742 60119 45840 33422 23223 15140 9094 5073 2605 1224 528 225 75 20 2");
}

TEST(TestFrontierSearch, Search6x2) {
	TestSearch<6, 2>(80, "1 2 3 6 11 20 36 60 95 155 258 426 688 1106 1723 2615 3901 5885 8851 13205 19508 28593 41179 58899 83582 118109 165136 228596 312542 423797 568233 755727 994641 1296097 1667002 2119476 2660415 3300586 4038877 4877286 5804505 6810858 7864146 8929585 9958080 10902749 11716813 12356080 12791679 13002649 12981651 12723430 12245198 11572814 10738102 9772472 8720063 7623133 6526376 5459196 4457799 3546306 2749552 2068975 1510134 1064591 720002 464913 284204 165094 89649 45758 21471 9583 3829 1427 430 129 33 12 2");
}

TEST(TestFrontierSearch, Search4x3) {
	TestSearch<4, 3>(53, "1 2 4 9 20 37 63 122 232 431 781 1392 2494 4442 7854 13899 24215 41802 71167 119888 198363 323206 515778 811000 1248011 1885279 2782396 4009722 5621354 7647872 10065800 12760413 15570786 18171606 20299876 21587248 21841159 20906905 18899357 16058335 12772603 9515217 6583181 4242753 2503873 1350268 643245 270303 92311 27116 5390 1115 86 18");
}

TEST(TestFrontierSearch, Search7x2) {
	STSearchOptions opts;
	opts.MaxDepth = 32;
	TestSearch<7, 2>(32, "1 2 3 6 11 20 37 67 117 198 329 557 942 1575 2597 4241 6724 10535 16396 25515 39362 60532 92089 138969 207274 307725 453000 664240 964874 1392975 1992353 2832063 3988528", opts);
}

TEST(TestFrontierSearch, Search5x3) {
	STSearchOptions opts;
	opts.MaxDepth = 23;
	TestSearch<5, 3>(23, "1 2 4 9 21 42 89 164 349 644 1349 2473 5109 9110 18489 32321 64962 112445 223153 378761 740095 1231589 2364342 3847629", opts);
}

TEST(TestFrontierSearch, Search8x2) {
	STSearchOptions opts;
	opts.MaxDepth = 25;
	TestSearch<8, 2>(25, "1 2 3 6 11 20 37 68 125 227 394 672 1151 1983 3373 5703 9508 15640 25293 40732 65032 103390 162830 255543 397013 613104", opts);
}

TEST(TestFrontierSearch, Search8x2_edge1) {
	STSearchOptions opts;
	opts.MaxDepth = 15;
	opts.InitialValue = "1 0 2 3 4 5 6 7  8 9 10 11 12 13 14 15";
	TestSearch<8, 2>(15, "1 3 5 8 15 29 52 95 173 302 518 902 1545 2629 4439 7446", opts);
}

TEST(TestFrontierSearch, Search8x2_edge2) {
	STSearchOptions opts;
	opts.MaxDepth = 15;
	opts.InitialValue = "1 2 0 3 4 5 6 7  8 9 10 11 12 13 14 15";
	TestSearch<8, 2>(15, "1 3 6 11 19 35 65 114 197 351 614 1056 1790 3040 5063 8375", opts);
}

TEST(TestFrontierSearch, Search8x2_edge3) {
	STSearchOptions opts;
	opts.MaxDepth = 15;
	opts.InitialValue = "1 2 3 0 4 5 6 7  8 9 10 11 12 13 14 15";
	TestSearch<8, 2>(15, "1 3 6 12 23 41 69 119 212 378 656 1139 1922 3219 5316 8776", opts);
}

TEST(TestFrontierSearch, Search4x4) {
	STSearchOptions opts;
	opts.MaxDepth = 20;
	TestSearch<4, 4>(20, "1 2 4 10 24 54 107 212 446 946 1948 3938 7808 15544 30821 60842 119000 231844 447342 859744 1637383", opts);
}

TEST(TestFrontierSearch, Search4x4_edge) {
	STSearchOptions opts;
	opts.MaxDepth = 15;
	opts.InitialValue = "1 0 2 3  4 5 6 7  8 9 10 11  12 13 14 15";
	TestSearch<4, 4>(15, "1 3 6 14 32 66 134 280 585 1214 2462 4946 9861 19600 38688 76086", opts);
}

TEST(TestFrontierSearch, Search4x4_center) {
	STSearchOptions opts;
	opts.MaxDepth = 15;
	opts.InitialValue = "1 2 3 4  5 0 6 7  8 9 10 11  12 13 14 15";
	TestSearch<4, 4>(15, "1 4 10 20 38 80 174 372 762 1540 3072 6196 12356 24516 48179 94356", opts);
}

