#include "pch.h"

#include "../SlidingTilePuzzle/GpuSolver.h"

TEST(TestGpuSolver, TestMoves4x4) {
    GpuSolver<4, 4> gpuSolver;
	std::string puzzle = "1 3 5 7 9 11 13 15 0 2 4 6 8 10 12 14";
	auto [segment, index] = gpuSolver.Rank(puzzle);
	HostBuffer segments, indexes;

	segments.Buffer[0] = segment;
	indexes.Buffer[0] = index;
	gpuSolver.GpuUp(indexes.Buffer, segments.Buffer, 1);

	auto puzzle2 = gpuSolver.Unrank(segments.Buffer[0], indexes.Buffer[0]);
	EXPECT_EQ(puzzle2, "1 3 5 7 0 11 13 15 9 2 4 6 8 10 12 14");
}

TEST(TestGpuSolver, TestMoves3x2) {
	GpuSolver<3, 2> gpuSolver;
	std::string puzzle = "3 1 2 5 0 4";
	auto [segment, index] = gpuSolver.Rank(puzzle);
	HostBuffer segments, indexes;

	segments.Buffer[0] = segment;
	indexes.Buffer[0] = index;
	gpuSolver.GpuUp(indexes.Buffer, segments.Buffer, 1);

	auto puzzle2 = gpuSolver.Unrank(segments.Buffer[0], indexes.Buffer[0]);
	EXPECT_EQ(puzzle2, "3 0 2 5 1 4");
}

TEST(TestGpuSolver, TestMoves5x3) {
	GpuSolver<5, 3> gpuSolver;
	std::string puzzle = "14 8 6 1 2 9 13 12 5 0 3 4 10 11 7";
	auto [segment, index] = gpuSolver.Rank(puzzle);
	HostBuffer segments, indexes;

	segments.Buffer[0] = segment;
	indexes.Buffer[0] = index;
	gpuSolver.GpuDown(indexes.Buffer, segments.Buffer, 1);

	auto puzzle2 = gpuSolver.Unrank(segments.Buffer[0], indexes.Buffer[0]);
	EXPECT_EQ(puzzle2, "14 8 6 1 2 9 13 12 5 7 3 4 10 11 0");
}

TEST(TestGpuSolverPerformance, TestMoves4x4) {
	GpuSolver<4, 4> gpuSolver;
	auto getVal = [&](int i) { return i * 16 + 5; };
	HostBuffer segments, indexes;
	for (int i = 0; i < indexes.SIZE; i++) {
		segments.Buffer[i] = 0;
		indexes.Buffer[i] = getVal(i);
	}
	gpuSolver.GpuUp(indexes.Buffer, segments.Buffer, indexes.SIZE);
	gpuSolver.GpuDown(indexes.Buffer, segments.Buffer, indexes.SIZE);
	for (int i = 0; i < indexes.SIZE; i++) {
		EXPECT_EQ(segments.Buffer[i], 0);
		EXPECT_EQ(indexes.Buffer[i], getVal(i));
	}
}

TEST(TestGpuSolverPerformance, TestMoves8x2) {
	GpuSolver<4, 4> gpuSolver;
	auto getVal = [&](int i) { return i * 16 + 5; };
	HostBuffer segments, indexes;
	for (int i = 0; i < indexes.SIZE; i++) {
		segments.Buffer[i] = 0;
		indexes.Buffer[i] = getVal(i);
	}
	gpuSolver.GpuUp(indexes.Buffer, segments.Buffer, indexes.SIZE);
	gpuSolver.GpuDown(indexes.Buffer, segments.Buffer, indexes.SIZE);
	for (int i = 0; i < indexes.SIZE; i++) {
		EXPECT_EQ(segments.Buffer[i], 0);
		EXPECT_EQ(indexes.Buffer[i], getVal(i));
	}
}
