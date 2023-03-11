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
#include "FrontierFile.h"
#include "Collector.h"
#include "Multiplexor.h"
#include "Puzzle.h"

template<int width, int height>
std::vector<uint64_t> FrontierSearch(SearchOptions options) {
	auto START = clock();
	Puzzle<width, height> puzzle;
	SegmentedFile frontier(puzzle.MaxSegments(), "d:/temp/frontier1");
	SegmentedFile new_frontier(puzzle.MaxSegments(), "d:/temp/frontier2");
	FrontierFileReader frontierReader(frontier);
	SegmentedFile e_up(puzzle.MaxSegments(), "d:/temp/expanded_up");
	SegmentedFile e_dn(puzzle.MaxSegments(), "d:/temp/expanded_dn");
	SegmentedFile e_lt(puzzle.MaxSegments(), "d:/temp/expanded_lt");
	SegmentedFile e_rt(puzzle.MaxSegments(), "d:/temp/expanded_rt");
	Collector collector(puzzle.MaxIndexesPerSegment(), new_frontier);
	Multiplexor m_up(puzzle.MaxSegments(), e_up);
	Multiplexor m_dn(puzzle.MaxSegments(), e_dn);
	Multiplexor m_lt(puzzle.MaxSegments(), e_lt);
	Multiplexor m_rt(puzzle.MaxSegments(), e_rt);
	ExpandedFrontierReader r_up(e_up);
	ExpandedFrontierReader r_dn(e_dn);
	ExpandedFrontierReader r_lt(e_lt);
	ExpandedFrontierReader r_rt(e_rt);

	auto initialIndex = puzzle.Rank(options.InitialValue);
	collector.SetSegment(initialIndex.first);
	collector.Add(initialIndex.second, puzzle.GetBounds(initialIndex.second));
	collector.SaveSegment();
	std::swap(frontier, new_frontier);

	std::vector<uint64_t> widths;
	widths.push_back(1);
	while (widths.size() <= options.MaxDepth) {

		std::cerr << "Depth: " << widths.size() - 1 << "; width: " << widths.back() << std::endl;

		// stage 1

		for (int segment = 0; segment < puzzle.MaxSegments(); segment++) {
			frontierReader.SetSegment(segment);
			while (true) {
				auto read = frontierReader.Read();
				if (read.Count == 0) break;
				for (size_t i = 0; i < read.Count; i++) {
					uint32_t index = read.Indexes[i];
					uint8_t bound = read.Bounds[i];
					if (!(bound & puzzle.B_UP)) {
						auto new_index = puzzle.MoveUp(segment, index);
						m_up.Add(new_index.first, new_index.second);
					}
					if (!(bound & puzzle.B_DOWN)) {
						auto new_index = puzzle.MoveDown(segment, index);
						m_dn.Add(new_index.first, new_index.second);
					}
					if (!(bound & puzzle.B_LEFT)) {
						auto new_index = puzzle.MoveLeft(index);
						m_lt.Add(segment, new_index);
					}
					if (!(bound & puzzle.B_RIGHT)) {
						auto new_index = puzzle.MoveRight(index);
						m_rt.Add(segment, new_index);
					}
				}
			}
		}
		m_up.Close();
		m_dn.Close();
		m_lt.Close();
		m_rt.Close();

		//stage 2

		size_t total = 0;

		for (int segment = 0; segment < puzzle.MaxSegments(); segment++) {
			r_up.SetSegment(segment);
			r_dn.SetSegment(segment);
			r_lt.SetSegment(segment);
			r_rt.SetSegment(segment);
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
				auto& buf = r_lt.Read();
				if (buf.Size() == 0) break;
				empty = false;
				for (size_t i = 0; i < buf.Size(); i++) {
					collector.Add(buf[i], puzzle.GetBounds(buf[i]) | puzzle.B_RIGHT);
				}
			}
			while (true) {
				auto& buf = r_rt.Read();
				if (buf.Size() == 0) break;
				empty = false;
				for (size_t i = 0; i < buf.Size(); i++) {
					collector.Add(buf[i], puzzle.GetBounds(buf[i]) | puzzle.B_LEFT);
				}
			}
			if (empty) continue;
			total += collector.SaveSegment();
			frontier.Delete(segment);
			e_up.Delete(segment);
			e_dn.Delete(segment);
			e_lt.Delete(segment);
			e_rt.Delete(segment);
		}
		if (total == 0) break;
		widths.push_back(total);
		frontier.DeleteAll();
		std::swap(frontier, new_frontier);
		e_up.DeleteAll();
		e_dn.DeleteAll();
		e_lt.DeleteAll();
		e_rt.DeleteAll();
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

