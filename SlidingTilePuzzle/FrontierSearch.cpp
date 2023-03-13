#include "GpuSolver.h"
#include "Puzzle.h"
#include "Util.h"

#include <chrono>
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
	Collector<width, height> collector(new_frontier);
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

	uint64_t timer_stage_1 = 0;
	uint64_t timer_stage_2 = 0;

	std::vector<uint64_t> widths;
	widths.push_back(1);

	std::cerr << "0: 1" << std::endl;

	while (widths.size() <= options.MaxDepth) {

		// stage 1

		Timer timerStartStep;

		auto frontierSize = frontier.TotalLength();

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
			verticalCollector.Close();
		}

		timer_stage_1 += timerStartStep.Elapsed();

		//stage 2

		auto expandedSize = e_up.TotalLength() + e_dn.TotalLength();

		Timer timerStartStage2;

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
				collector.AddUpMoves(buf.Buf(), buf.Size());
			}
			while (true) {
				auto& buf = r_dn.Read();
				if (buf.Size() == 0) break;
				empty = false;
				collector.AddDownMoves(buf.Buf(), buf.Size());
			}
			while (true) {
				auto buf = frontierReader.Read();
				if (buf.Count == 0) break;
				empty = false;
				collector.AddHorizontalMoves(buf.Indexes, buf.Bounds, buf.Count);
			}
			if (empty) continue;
			total += collector.SaveSegment();
			frontier.Delete(segment);
			e_up.Delete(segment);
			e_dn.Delete(segment);
		}
		if (total == 0) break;

		timer_stage_2 += timerStartStage2.Elapsed();

		auto newFrontierSize = new_frontier.TotalLength();

		std::cerr
			<< widths.size() << ": " << WithDecSep(total)
			<< " time=" << WithTime(timerStartStep.Elapsed())
			<< " Files=" << WithSize(frontierSize) 
			<< ", " << WithSize(expandedSize)
			<< ", " << WithSize(newFrontierSize)
			<< std::endl;

		widths.push_back(total);
		frontier.DeleteAll();
		std::swap(frontier, new_frontier);
		e_up.DeleteAll();
		e_dn.DeleteAll();
	}
	 
	auto FINISH = clock();
	std::cerr << "Finished in " << WithDecSep(FINISH - START) << std::endl;

	std::cerr << " stage 1: " << WithTime(timer_stage_1) << std::endl;
	std::cerr << " stage 2: " << WithTime(timer_stage_2) << std::endl;

	collector.PrintStats();
	GpuSolver<width, height>::PrintStats();
	SegmentedFile::PrintStats();

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

