#include "../SlidingTilePuzzle/GpuSolver.h"
#include "../SlidingTilePuzzle/Puzzle.h"
#include "../SlidingTilePuzzle/Util.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <time.h>
#include <vector>

int main() {
	auto START = clock();

	Puzzle<4, 3> puzzle;
	static_assert(puzzle.MaxSegments() == 1);
	constexpr int SIZE = puzzle.MaxIndexesPerSegment();

	std::vector<bool> front1(SIZE);
	std::vector<bool> front2(SIZE);
	std::vector<bool> visited(SIZE);

	auto& front = front1;
	auto& new_front = front2;

	for (int i = 0; i < SIZE; i++) {
		front[i] = false;
		visited[i] = false;
		new_front[i] = false;
	}
	auto initial = puzzle.Rank("0 1 2 3 4 5").second;
	front[initial] = true;
	visited[initial] = true;

	std::vector<uint64_t> widths;
	widths.push_back(1);

	while (true) {
		std::cerr << widths.size() - 1 << ": " << WithDecSep(widths.back()) << std::endl;

		uint64_t count = 0;

		auto addIdx = [&](int idx) {
			if (!visited[idx]) {
				visited[idx] = true;
				new_front[idx] = true;
				count++;
			}
		};

		for (int i = 0; i < SIZE; i++) {
			if (!front[i]) continue;
			if (puzzle.CanMoveUp(i)) 
				addIdx(puzzle.MoveUp(0, i).second);
			if (puzzle.CanMoveDown(i))
				addIdx(puzzle.MoveDown(0, i).second);
			if (puzzle.CanMoveLeft(i))
				addIdx(puzzle.MoveLeft(i));
			if (puzzle.CanMoveRight(i))
				addIdx(puzzle.MoveRight(i));
		}

		if (count == 0) break;
		widths.push_back(count);
		std::swap(front, new_front);
		for (int i = 0; i < SIZE; i++) new_front[i] = false;
	}

	auto FINISH = clock();
	std::cerr << "finished in " << WithDecSep(FINISH - START) << std::endl;
}
