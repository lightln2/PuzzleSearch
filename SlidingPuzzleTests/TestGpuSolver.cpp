#include "pch.h"

#include "../SlidingTilePuzzle/GpuSolver.h"
#include "../SlidingTilePuzzle/Puzzle.h"
#include "../SlidingTilePuzzle/Util.h"

template<int width, int height>
void TestMove(std::string puzzleStr) {
	Puzzle<width, height> puzzle;
	auto stateHash = puzzle.Rank(puzzleStr);

	GpuSolver<width, height> gpuSolver;
	HostBuffer segments, indexes;

	if (puzzle.CanMoveUp(stateHash.second)) {
		auto expected = puzzle.MoveUp(stateHash);
		segments.Buffer[0] = stateHash.first;
		indexes.Buffer[0] = stateHash.second;
		gpuSolver.GpuUp(stateHash.first, indexes.Buffer, segments.Buffer, 1);
		EXPECT_EQ(segments.Buffer[0], expected.first);
		EXPECT_EQ(indexes.Buffer[0], expected.second);
	}

	if (puzzle.CanMoveDown(stateHash.second)) {
		auto expected = puzzle.MoveDown(stateHash);
		segments.Buffer[0] = stateHash.first;
		indexes.Buffer[0] = stateHash.second;
		gpuSolver.GpuDown(stateHash.first, indexes.Buffer, segments.Buffer, 1);
		EXPECT_EQ(segments.Buffer[0], expected.first);
		EXPECT_EQ(indexes.Buffer[0], expected.second);
	}

}

TEST(TestGpuSolver, TestMoves4x4) {
	TestMove<4, 4>("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15");
	TestMove<4, 4>("15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0");
	TestMove<4, 4>("15 14 8 6 1 2 9 13 12 5 0 3 4 10 11 7");
}

TEST(TestGpuSolverPerformance, TestPerformance4x4) {
	Puzzle<4, 4> puzzle;
	GpuSolver<4, 4> gpuSolver;
	HostBuffer segmentsUp, segmentsDown, indexesUp, indexesDown;
	int posUp = 0, posDown = 0;

	constexpr int MAX = 10 * 1000 * 1000;
	uint64_t hashUp = 0, hashDown = 0;

	constexpr int SEGMENT = 0x5a2;

	auto consumeUp = [&]() {
		gpuSolver.GpuUp(SEGMENT, indexesUp.Buffer, segmentsUp.Buffer, posUp);
		for (int i = 0; i < posUp; i++) {
			auto segment = segmentsUp.Buffer[i];
			auto index = indexesUp.Buffer[i];
			auto blank = index % 16;
			hashUp = hashUp * 31 + (uint64_t(segment) << 32) + index;
			ENSURE_EQ(SEGMENT != segment, blank == 9 || blank == 10 || blank == 11);
		}
		posUp = 0;
	};

	auto consumeDown = [&]() {
		gpuSolver.GpuDown(SEGMENT, indexesDown.Buffer, segmentsDown.Buffer, posDown);
		for (int i = 0; i < posDown; i++) {
			auto segment = segmentsDown.Buffer[i];
			auto index = indexesDown.Buffer[i];
			auto blank = index % 16;
			hashDown = hashDown * 31 + (uint64_t(segment) << 32) + index;
			ENSURE_EQ(SEGMENT != segment, blank == 13 || blank == 14 || blank == 15);
		}
		posDown = 0;
	};

	for (uint32_t i = 0; i < MAX; i++) {
		int blank = i % 16;
		if (puzzle.CanMoveUp(i)) {
			segmentsUp.Buffer[posUp] = SEGMENT;
			indexesUp.Buffer[posUp++] = i;
			if (posUp == indexesUp.SIZE) consumeUp();
		}
		if (puzzle.CanMoveDown(i)) {
			segmentsDown.Buffer[posDown] = SEGMENT;
			indexesDown.Buffer[posDown++] = i;
			if (posDown == indexesDown.SIZE) consumeDown();
		}
	}
	consumeUp();
	consumeDown();

	EXPECT_EQ(hashUp, 15956298610708895712);
	EXPECT_EQ(hashDown, 8254655271410123088);
}

TEST(TestGpuSolverPerformance, TestMoves4x4) {
	GpuSolver<4, 4> gpuSolver;
	constexpr uint32_t SEGMENT = 0;
	auto getVal = [&](int i) { return i * 16 + 5; };

	HostBuffer segments, indexes;
	for (int i = 0; i < indexes.SIZE; i++) {
		indexes.Buffer[i] = getVal(i);
	}
	gpuSolver.GpuUp(SEGMENT, indexes.Buffer, segments.Buffer, indexes.SIZE);
	gpuSolver.GpuDown(segments.Buffer[0], indexes.Buffer, segments.Buffer, indexes.SIZE);
	for (int i = 0; i < indexes.SIZE; i++) {
		ENSURE_EQ(segments.Buffer[i], SEGMENT);
		ENSURE_EQ(indexes.Buffer[i], getVal(i));
	}
}

TEST(TestGpuSolverPerformance, TestMoves8x2) {
	GpuSolver<8, 2> gpuSolver;
	constexpr uint32_t SEGMENT = 0;
	auto getVal = [&](uint32_t i) { return i * 16 + 5; };
	HostBuffer segments, indexes;
	for (int i = 0; i < indexes.SIZE; i++) {
		indexes.Buffer[i] = getVal(i);
	}
	gpuSolver.GpuDown(SEGMENT, indexes.Buffer, segments.Buffer, indexes.SIZE);
	gpuSolver.GpuUp(segments.Buffer[0], indexes.Buffer, segments.Buffer, indexes.SIZE);
	for (int i = 0; i < indexes.SIZE; i++) {
		ENSURE_EQ(segments.Buffer[i], SEGMENT);
		ENSURE_EQ(indexes.Buffer[i], getVal(i));
	}
}

