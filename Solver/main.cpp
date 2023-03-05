#include "../SlidingTilePuzzle/GpuSolver.h"
#include "../SlidingTilePuzzle/Puzzle.h"
#include "../SlidingTilePuzzle/Util.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <time.h>

int main() {

	auto START = clock();
	Puzzle<4, 4> puzzle;
	GpuSolver<4, 4> gpuSolver;
	HostBuffer segmentsUp, segmentsDown, indexesUp, indexesDown;
	int posUp = 0, posDown = 0;

	constexpr int MAX = 100 * 1000 * 1000;
	uint64_t hashUp = 0, hashDown = 0;

	constexpr int SEGMENT = 0x5a2;

	auto consumeUp = [&]() {
		gpuSolver.GpuUp(indexesUp.Buffer, segmentsUp.Buffer, posUp);
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
		gpuSolver.GpuDown(indexesDown.Buffer, segmentsDown.Buffer, posDown);
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

	//ENSURE_EQ(hashUp, 15956298610708895712);
	//ENSURE_EQ(hashDown, 8254655271410123088);

	auto END = clock();
	std::cerr << "Hash up: " << hashUp << "; down: " << hashDown << std::endl;
	std::cerr << "Time: " << END - START << std::endl;

	/*
	auto START = clock();
	Puzzle<4, 4> puzzle;
	constexpr int MAX = 10 * 1000 * 1000;

	int upMoves[4096]{ 0 };
	int downMoves[4096]{ 0 };
	int totalUp = 0, totalDown = 0;
	uint64_t hashUp = 0, hashDown = 0;

	int SEGMENT = 0x5a2;

	for (uint32_t i = 0; i < MAX; i++) {
		int blank = i % 16;
		if (puzzle.CanMoveUp(i)) {
			auto res = puzzle.MoveUp(SEGMENT, i);
			upMoves[res.first]++;
			totalUp++;
			hashUp = hashUp * 31 + (uint64_t(res.first) << 32) + res.second;
			ensure((SEGMENT != res.first) == (blank == 13 || blank == 14 || blank == 15));
		}
		if (puzzle.CanMoveDown(i)) {
			auto res = puzzle.MoveDown(SEGMENT, i);
			downMoves[res.first]++;
			totalDown++;
			hashDown = hashDown * 31 + (uint64_t(res.first) << 32) + res.second;
			ensure((SEGMENT != res.first) == (blank == 9 || blank == 10 || blank == 11));
		}
	}

	auto END = clock();
	for (int i = 0; i < 4096; i++) {
		if (upMoves[i] > 0)
			std::cerr << "up[" << std::setw(3) << std::hex << i << std::dec << "]: " << upMoves[i] << std::endl;
	}
	for (int i = 0; i < 4096; i++) {
		if (downMoves[i] > 0)
			std::cerr << "down[" << std::setw(3) << std::hex << i << std::dec << "]: " << downMoves[i] << std::endl;
	}
	std::cerr << "Total up: " << totalUp << "; down: " << totalDown << std::endl;
	std::cerr << "Hash up: " << hashUp << "; down: " << hashDown << std::endl;
	std::cerr << "Time: " << END - START << std::endl;
	*/

	/*
	GpuSolver<4, 4> gpuSolver;

	HostBuffer indexes, segments;

	constexpr uint32_t SEG = 1259;
	constexpr uint32_t MAX = 10 * 1000 * 1000;

	uint32_t totalMoves = 0;

	size_t pos = 0;

	uint64_t segarr[5000]{ 0 };

	auto START = clock();
	for (uint32_t i = 0; i < MAX; i++) {
		if (Puzzle<4, 4>::CanMoveUp(i)) {
			segments.Buffer[pos++] = SEG;
			indexes.Buffer[pos++] = i;
			if (pos == indexes.SIZE) {
				gpuSolver.GpuUp(indexes.Buffer, segments.Buffer, pos);
				for (int j = 0; j < pos; j++) {
					segarr[segments.Buffer[j]]++;
				}
				totalMoves += pos;
				pos = 0;
			}
		}
	}
	if (pos > 0) {
		gpuSolver.GpuUp(indexes.Buffer, segments.Buffer, pos);
		for (int j = 0; j < pos; j++) {
			segarr[segments.Buffer[j]]++;
		}
		totalMoves += pos;
		pos = 0;
	}
	auto END = clock();

	int totalSegs = 0;
	for (int i = 0; i < 5000; i++) {
		if (segarr[i] > 0) {
			totalSegs++;
			std::cerr << i << ": " << segarr[i] <<  " (" << (segarr[i] * 10000 / totalMoves) << "..%)" << std::endl;
		}
	}
	std::cerr << "Processed " << totalMoves << " into " << totalSegs << " segments in " << END - START << std::endl;
	return 0;
	*/
}
