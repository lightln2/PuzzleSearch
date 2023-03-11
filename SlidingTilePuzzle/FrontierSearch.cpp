#include "GpuSolver.h"
#include "Puzzle.h"
#include "Util.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <time.h>
#include <vector>

#include "Collector.h"
#include "FrontierFile.h"
#include "FrontierSearch.h"
#include "Multiplexor.h"
#include "Puzzle.h"
#include "SegmentedFile.h"
#include "VerticalMovesCollector.h"

template<int width, int height>
std::vector<uint64_t> FrontierSearch(SearchOptions options) {
	auto START = clock();
	Puzzle<width, height> puzzle;
	SegmentedFile frontier(puzzle.MaxSegments(), "d:/temp/frontier1");
	SegmentedFile new_frontier(puzzle.MaxSegments(), "d:/temp/frontier2");
	FrontierFileReader frontierReader(frontier);
	SegmentedFile e_up(puzzle.MaxSegments(), "d:/temp/expanded_up");
	SegmentedFile e_dn(puzzle.MaxSegments(), "d:/temp/expanded_dn");
	Collector collector(puzzle.MaxIndexesPerSegment(), new_frontier);
	VerticalMovesCollector<width, height> verticalCollector(e_up, e_dn);
	ExpandedFrontierReader r_up(e_up);
	ExpandedFrontierReader r_dn(e_dn);

	{
		auto [initialSegment, initialIndex] = puzzle.Rank(options.InitialValue);
		FrontierFileWriter fwriter(frontier);
		fwriter.SetSegment(initialSegment);
		fwriter.Add(initialIndex, puzzle.GetBounds(initialIndex));
		fwriter.FinishSegment();
	}

	std::vector<uint64_t> widths;
	widths.push_back(1);
	while (widths.size() <= options.MaxDepth) {

		std::cerr << "Depth: " << widths.size() - 1 << "; width: " << widths.back() << std::endl;

		// stage 1

		for (int segment = 0; segment < puzzle.MaxSegments(); segment++) {
			frontierReader.SetSegment(segment);
			verticalCollector.SetSegment(segment);
			while (true) {
				auto read = frontierReader.Read();
				if (read.Count == 0) break;
				for (size_t i = 0; i < read.Count; i++) {
					uint32_t index = read.Indexes[i];
					uint8_t bound = read.Bounds[i];
					if (!(bound & puzzle.B_UP)) {
						verticalCollector.AddUp(index);
					}
					if (!(bound & puzzle.B_DOWN)) {
						verticalCollector.AddDown(index);
					}
				}
			}
		}
		verticalCollector.Close();

		//stage 2

		size_t total = 0;

		for (int segment = 0; segment < puzzle.MaxSegments(); segment++) {
			r_up.SetSegment(segment);
			r_dn.SetSegment(segment);
			frontierReader.SetSegment(segment);
			collector.SetSegment(segment);

			bool empty = true;
			while (true) {
				auto& buf = r_up.Read();
				if (buf.Size() == 0) break;
				empty = false;
				for (size_t i = 0; i < buf.Size(); i++) {
					collector.Add(buf[i], puzzle.GetBounds(buf[i]) | puzzle.B_DOWN);
				}
			}
			while (true) {
				auto& buf = r_dn.Read();
				if (buf.Size() == 0) break;
				empty = false;
				for (size_t i = 0; i < buf.Size(); i++) {
					collector.Add(buf[i], puzzle.GetBounds(buf[i]) | puzzle.B_UP);
				}
			}
			while (true) {
				auto buf = frontierReader.Read();
				if (buf.Count == 0) break;
				empty = false;
				for (size_t i = 0; i < buf.Count; i++) {
					uint32_t index = buf.Indexes[i];
					uint8_t bound = buf.Bounds[i];
					if (!(bound & puzzle.B_LEFT)) {
						auto leftMove = puzzle.MoveLeft(index);
						collector.Add(leftMove, puzzle.GetBounds(leftMove) | puzzle.B_RIGHT);
					}
					if (!(bound & puzzle.B_RIGHT)) {
						auto rightMove = puzzle.MoveRight(index);
						collector.Add(rightMove, puzzle.GetBounds(rightMove) | puzzle.B_LEFT);
					}
				}
			}
			if (empty) continue;
			total += collector.SaveSegment();
			frontier.Delete(segment);
			e_up.Delete(segment);
			e_dn.Delete(segment);
		}
		if (total == 0) break;
		widths.push_back(total);
		frontier.DeleteAll();
		std::swap(frontier, new_frontier);
		e_up.DeleteAll();
		e_dn.DeleteAll();
	}
	 
	auto FINISH = clock();
	std::cerr << "Finished in " << WithDecSep(FINISH - START) << std::endl;
	return widths;
}

template std::vector<uint64_t> FrontierSearch<2, 2>(SearchOptions options);
template std::vector<uint64_t> FrontierSearch<3, 2>(SearchOptions options);
template std::vector<uint64_t> FrontierSearch<4, 2>(SearchOptions options);
template std::vector<uint64_t> FrontierSearch<5, 2>(SearchOptions options);
template std::vector<uint64_t> FrontierSearch<6, 2>(SearchOptions options);
template std::vector<uint64_t> FrontierSearch<7, 2>(SearchOptions options);
template std::vector<uint64_t> FrontierSearch<8, 2>(SearchOptions options);

template std::vector<uint64_t> FrontierSearch<3, 3>(SearchOptions options);
template std::vector<uint64_t> FrontierSearch<4, 3>(SearchOptions options);
template std::vector<uint64_t> FrontierSearch<5, 3>(SearchOptions options);

template std::vector<uint64_t> FrontierSearch<4, 4>(SearchOptions options);

