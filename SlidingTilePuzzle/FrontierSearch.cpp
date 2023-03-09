#include "GpuSolver.h"
#include "Puzzle.h"
#include "Util.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <time.h>
#include <vector>
#include <fstream>

#include "FrontierSearch.h"
#include "SegmentedFile.h"
#include "Puzzle.h"

void ClassicFrontierSearch() {
	constexpr size_t BUFFER_SIZE = 4 * 1024 * 1024;
	constexpr int B_UP = 1;
	constexpr int B_DOWN = 2;
	constexpr int B_LEFT = 4;
	constexpr int B_RIGHT = 8;

	auto START = clock();
	Puzzle<4, 3> puzzle;
	SegmentedFile frontier1(puzzle.MaxSegments(), "d:/temp/frontier1");
	SegmentedFile frontier2(puzzle.MaxSegments(), "d:/temp/frontier2");
	auto& frontier = frontier1;
	auto& new_frontier = frontier2;

	auto getBounds = [&](uint32_t index) {
		uint32_t bounds = 0;
		if (puzzle.CanMoveUp(index)) bounds |= B_UP;
		if (puzzle.CanMoveDown(index)) bounds |= B_DOWN;
		if (puzzle.CanMoveLeft(index)) bounds |= B_LEFT;
		if (puzzle.CanMoveRight(index)) bounds |= B_RIGHT;
		return bounds;
	};

	std::vector<uint32_t> buffer(BUFFER_SIZE);
	uint32_t* buf = &buffer[0];
	size_t pos = 0;

	std::vector<uint32_t> buffer2(BUFFER_SIZE);
	uint32_t* buf2 = &buffer2[0];
	size_t pos2 = 0;

	auto initialIndex = puzzle.Rank("0 1 2 3 4 5 6 7 8 9 10 11");
	buf[pos++] = initialIndex.second;
	buf[pos++] = getBounds(initialIndex.second);
	frontier.Write(initialIndex.first, buf, pos * 4);
	pos = 0;

	std::vector<uint64_t> widths;
	widths.push_back(1);

	std::vector<uint8_t> collector(puzzle.MaxSegments());

	while (true) {

		std::cerr << "Depth: " << widths.size() - 1 << "; width: " << widths.back() << std::endl;

		// stage 1
		for (int segment = 0; segment < puzzle.MaxSegments(); segment++) {
			memset(&collector[0], 0, collector.size());
			while (true) {
				auto read = frontier.Read(segment, buf, BUFFER_SIZE) / 8;
				if (read == 0) break;
				pos = 0;
				pos2 = 0;
				for (int i = 0; i < read; i++) {
					uint32_t index = buf[pos++];
					uint32_t bounds = buf[pos++];
					if (!(bounds & B_UP)) {
						if (puzzle.CanMoveUp(index)) {
							auto newindex = puzzle.MoveUp(segment, index);
							ensure(newindex.first == segment);
							buf2[pos2++] = newindex.second;
							buf2[pos2++] = getBounds(newindex.second) | B_DOWN;
						}
					}
				}
			}

		}

	}
	

	int depth = 0;

	auto FINISH = clock();
	std::cerr << "Finished in " << WithDecSep(START - FINISH) << std::endl;

}

