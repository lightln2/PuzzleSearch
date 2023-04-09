#include "pch.h"

#include "../SlidingTilePuzzle/MTFrontierSearch.h"
#include "../SlidingTilePuzzle/Puzzle.h"

template<int width, int height>
void TestSearch(size_t expected_radius, std::string counts, MTSearchOptions options = {}) {
	auto result = MTFrontierSearch<width, height>(options);
	EXPECT_EQ(result.size() - 1, expected_radius);

	std::ostringstream stream;
	for (size_t i = 0; i < result.size(); i++) {
		if (i > 0) stream << ' ';
		stream << result[i];
	}
	EXPECT_EQ((stream.str()), counts);
}

TEST(TestMTFrontierSearch, Search2x2) {
	TestSearch<2, 2>(6, "1 2 2 2 2 2 1");
}

TEST(TestMTFrontierSearch, Search3x2) {
	TestSearch<3, 2>(20, "1 3 4 6 8 12 15 18 22 30 36 45 42 30 28 24 18 9 5 3 1");
}

TEST(TestMTFrontierSearch, Search3x2_edge) {
	MTSearchOptions opts;
	opts.InitialValue = "1 0 2  3 4 5";
	TestSearch<3, 2>(20, "1 3 4 6 8 12 14 18 24 30 36 45 43 30 24 24 20 9 5 3 1", opts);
}

TEST(TestMTFrontierSearch, Search4x2) {
	TestSearch<4, 2>(25,
		"1 4 6 12 18 36 53 96 136 232 324 544 728 1064 1366 1928 2321 2780 2884 2436 1825 800 368 140 50 8");
}

TEST(TestMTFrontierSearch, Search4x2_edge) {
	MTSearchOptions opts;
	opts.InitialValue = "1 0 2 3  4 5 6 7";
	TestSearch<4, 2>(26,
		"1 4 6 12 18 36 52 96 138 232 322 544 723 1064 1375 1928 2343 2780 2837 2436 1821 800 391 140 50 8 3", opts);
}

TEST(TestMTFrontierSearch, Search3x3) {
	TestSearch<3, 3>(24,
		"1 4 8 16 32 64 127 244 454 856 1576 2854 5117 8588 13466 19739 26558 31485 30985 23494 11751 3390 589 41 1");
}

TEST(TestMTFrontierSearch, Search3x3_edge) {
	MTSearchOptions opts;
	opts.InitialValue = "1 0 2  3 4 5  6 7 8";
	TestSearch<3, 3>(24,
		"1 4 8 16 32 64 126 238 456 862 1590 2863 5114 8618 13449 19642 26478 31502 30815 23401 12148 3381 588 42 2", opts);
}

TEST(TestMTFrontierSearch, Search3x3_center) {
	MTSearchOptions opts;
	opts.InitialValue = "1 2 3  4 0 5  6 7 8";
	TestSearch<3, 3>(24,
		"1 4 8 16 32 64 124 236 452 872 1598 2880 5048 8632 13500 19412 26136 31859 30304 23402 12743 3424 654 35 4", opts);
}

TEST(TestMTFrontierSearch, Search5x2) {
	TestSearch<5, 2>(36,
		"1 5 8 20 32 80 127 300 458 960 1458 3055 4540 8780 12694 23520 33370 57200 77373 119315 151791 204225 235617 256305 246642 178775 120853 49655 20885 4530 1229 440 113 30 8 5 1");
}

TEST(TestMTFrontierSearch, Search5x2_edge1) {
	MTSearchOptions opts;
	opts.InitialValue = "1 0 2 3 4  5 6 7 8 9";
	TestSearch<5, 2>(36,
		"1 5 8 20 32 80 126 300 460 960 1452 3055 4535 8780 12713 23520 33327 57200 77109 119315 151662 204225 235561 256305 245483 178775 121302 49655 21776 4530 1520 440 123 30 9 5 1", opts);
}

TEST(TestMTFrontierSearch, Search5x2_edge2) {
	MTSearchOptions opts;
	opts.InitialValue = "1 2 0 3 4  5 6 7 8 9";
	TestSearch<5, 2>(36,
		"1 5 8 20 32 80 126 300 460 960 1448 3055 4545 8780 12750 23520 33058 57200 76893 119315 150211 204225 235212 256305 245994 178775 122405 49655 22293 4530 1633 440 122 30 8 5 1", opts);
}

TEST(TestMTFrontierSearch, Search6x2) {
	TestSearch<6, 2>(41,
		"1 6 10 30 50 150 249 720 1152 2880 4610 11544 18128 42684 65924 149514 228778 488796 728433 1467000 2136054 3978876 5574328 9510150 12699920 19077354 23527929 29862012 32688655 31474098 27976007 18457764 12016184 4687338 1939160 519318 141371 19092 3257 1038 200 36");
}

TEST(TestMTFrontierSearch, Search4x3) {
	TestSearch<4, 3>(33,
		"1 5 12 30 72 180 431 1058 2418 5711 12858 29630 65053 145090 303771 640141 1260032 2476812 4490822 7999853 12981931 20265326 28065825 36086638 39470660 37460934 27258384 14584710 4854329 939300 93345 5229 204 5");
}

TEST(TestMTFrontierSearch, Search7x2) {
	MTSearchOptions opts;
	opts.MaxDepth = 23;
	TestSearch<7, 2>(23,
		"1 7 12 42 72 252 431 1470 2430 7070 11728 34125 55550 153188 247010 659302 1055500 2672600 4206121 10228379 15906111 36437156 55403239 120204434", opts);
}

TEST(TestMTFrontierSearch, Search5x3) {
	MTSearchOptions opts;

	//opts.MaxDepth = 20;
	//TestSearch<5, 3>(20,
	//	"1 6 16 48 128 384 1023 3036 7796 22155 55915 155931 385558 1049703 2524357 6646590 15411981 38957481 86507301 207621178 436566702", opts);
	opts.MaxDepth = 17;
	TestSearch<5, 3>(17,
		"1 6 16 48 128 384 1023 3036 7796 22155 55915 155931 385558 1049703 2524357 6646590 15411981 38957481", opts);
}

TEST(TestMTFrontierSearch, Search8x2) {
	MTSearchOptions opts;
	opts.MaxDepth = 15;
	TestSearch<8, 2>(15,
		"1 8 14 56 98 392 685 2688 4552 15120 25736 85280 142472 449216 746786 2286992", opts);
}

TEST(TestMTFrontierSearch, Search8x2_edge1) {
	MTSearchOptions opts;
	opts.MaxDepth = 15;
	opts.InitialValue = "1 0 2 3 4 5 6 7  8 9 10 11 12 13 14 15";
	TestSearch<8, 2>(15,
		"1 8 14 56 98 392 684 2688 4554 15120 25694 85280 142491 449216 746323 2286992", opts);
}

TEST(TestMTFrontierSearch, Search8x2_edge2) {
	MTSearchOptions opts;
	opts.MaxDepth = 15;
	opts.InitialValue = "1 2 0 3 4 5 6 7  8 9 10 11 12 13 14 15";
	TestSearch<8, 2>(15,
		"1 8 14 56 98 392 684 2688 4554 15120 25636 85280 143055 449216 746452 2286992", opts);
}

TEST(TestMTFrontierSearch, Search8x2_edge3) {
	MTSearchOptions opts;
	opts.MaxDepth = 15;
	opts.InitialValue = "1 2 3 0 4 5 6 7  8 9 10 11 12 13 14 15";
	TestSearch<8, 2>(15,
		"1 8 14 56 98 392 684 2688 4554 15120 25584 85280 143559 449216 746592 2286992", opts);
}

TEST(TestMTFrontierSearch, Search4x4) {
	MTSearchOptions opts;
	opts.FileExpanded1 = { "c:/temp/frontierExp1.part1", "c:/temp/frontierExp1.part2", "c:/temp/frontierExp1.part3" };
	opts.FileExpanded2 = { "c:/temp/frontierExp2.part1", "c:/temp/frontierExp2.part2", "c:/temp/frontierExp2.part3" };
	opts.ExpandedFileSequentialParts = true;
	opts.MaxDepth = 15;
	TestSearch<4, 4>(15,
		"1 6 18 54 162 486 1457 4334 12568 36046 102801 289534 808623 2231878 6076994 16288752", opts);
}

TEST(TestMTFrontierSearch, Search4x4_edge) {
	MTSearchOptions opts;
	opts.MaxDepth = 15;
	opts.InitialValue = "1 2 3 4  0 5 6 7  8 9 10 11  12 13 14 15";
	TestSearch<4, 4>(15,
		"1 6 18 54 162 486 1456 4312 12522 35932 102302 288084 804468 2219605 6039488 16167649", opts);
}

TEST(TestMTFrontierSearch, Search4x4_center) {
	MTSearchOptions opts;
	opts.MaxDepth = 15;
	opts.InitialValue = "1 2 3 4  5 0 6 7  8 9 10 11  12 13 14 15";
	TestSearch<4, 4>(15,
		"1 6 18 54 162 486 1454 4300 12458 35850 101749 286640 799588 2206318 5995157 16026648", opts);
}
