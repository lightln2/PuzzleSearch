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

void ClassicFrontierSearch() {
	constexpr size_t BUFFER_SIZE = 4 * 1024 * 1024;
	constexpr int B_UP = 1;
	constexpr int B_DOWN = 2;
	constexpr int B_LEFT = 4;
	constexpr int B_RIGHT = 8;

	auto START = clock();
	Puzzle<4, 3> puzzle;
	SegmentedFile file_frontier1(puzzle.MaxSegments(), "d:/temp/frontier1");
	SegmentedFile file_frontier2(puzzle.MaxSegments(), "d:/temp/frontier2");
	SegmentedFile e_up(puzzle.MaxSegments(), "d:/temp/expanded_up");
	SegmentedFile e_dn(puzzle.MaxSegments(), "d:/temp/expanded_dn");
	SegmentedFile e_lt(puzzle.MaxSegments(), "d:/temp/expanded_lt");
	SegmentedFile e_rt(puzzle.MaxSegments(), "d:/temp/expanded_rt");
	FrontierFileReader frontierReader;
	FrontierFileWriter frontierWriter;
	ExpandedFrontierWriter w_up, w_dn, w_lt, w_rt;
	ExpandedFrontierReader r_up, r_dn, r_lt, r_rt;
	SegmentedFile* frontier = &file_frontier1;
	SegmentedFile* new_frontier = &file_frontier2;
	Collector collector(puzzle.MaxIndexesPerSegment(), frontierWriter);
	Multiplexor m_up(puzzle.MaxSegments(), &e_up, w_up);
	Multiplexor m_dn(puzzle.MaxSegments(), &e_dn, w_dn);
	Multiplexor m_lt(puzzle.MaxSegments(), &e_lt, w_lt);
	Multiplexor m_rt(puzzle.MaxSegments(), &e_rt, w_rt);

	Buffer<uint32_t> buffer(BUFFER_SIZE);
	Buffer<uint32_t> buffer2(BUFFER_SIZE);

	auto initialIndex = puzzle.Rank("0 1 2 3 4 5 6 7 8 9 10 11");
	frontierWriter.SetSegment(frontier, initialIndex.first);
	frontierWriter.Add(initialIndex.second, puzzle.GetBounds(initialIndex.second));
	frontierWriter.FinishSegment();

	std::vector<uint64_t> widths;
	widths.push_back(1);

	while (true) {

		std::cerr << "Depth: " << widths.size() - 1 << "; width: " << widths.back() << std::endl;

		// stage 1
		for (int segment = 0; segment < puzzle.MaxSegments(); segment++) {
			frontierReader.SetSegment(frontier, segment);
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
			r_up.SetSegment(&e_up, segment);
			r_dn.SetSegment(&e_dn, segment);
			r_lt.SetSegment(&e_lt, segment);
			r_rt.SetSegment(&e_rt, segment);
			frontierWriter.SetSegment(new_frontier, segment);
			collector.SetSegment(segment);

			while (true) {
				auto& buf = r_up.Read();
				if (buf.Size() == 0) break;
				for (size_t i = 0; i < buf.Size(); i++) {
					collector.Add(buf[i], puzzle.GetBounds(buf[i]) | puzzle.B_DOWN);
				}
			}
			while (true) {
				auto& buf = r_dn.Read();
				if (buf.Size() == 0) break;
				for (size_t i = 0; i < buf.Size(); i++) {
					collector.Add(buf[i], puzzle.GetBounds(buf[i]) | puzzle.B_UP);
				}
			}
			while (true) {
				auto& buf = r_lt.Read();
				if (buf.Size() == 0) break;
				for (size_t i = 0; i < buf.Size(); i++) {
					collector.Add(buf[i], puzzle.GetBounds(buf[i]) | puzzle.B_RIGHT);
				}
			}
			while (true) {
				auto& buf = r_rt.Read();
				if (buf.Size() == 0) break;
				for (size_t i = 0; i < buf.Size(); i++) {
					collector.Add(buf[i], puzzle.GetBounds(buf[i]) | puzzle.B_LEFT);
				}
			}
			total += collector.SaveSegment();
		}
		if (total == 0) break;
		widths.push_back(total);
		std::swap(frontier, new_frontier);
		new_frontier->DeleteAll();
		e_up.DeleteAll();
		e_dn.DeleteAll();
		e_lt.DeleteAll();
		e_rt.DeleteAll();
	}
	
	auto FINISH = clock();
	std::cerr << "Finished in " << WithDecSep(FINISH - START) << std::endl;

}

