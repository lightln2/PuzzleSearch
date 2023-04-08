#include "GpuSolver.h"
#include "Puzzle.h"
#include "Util.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <time.h>
#include <vector>

#include "MTCollector.h"
#include "MTFrontierFile.h"
#include "MTFrontierSearch.h"
#include "MTVerticalMovesCollector.h"
#include "Multiplexor.h"
#include "Puzzle.h"
#include "SegmentedFile.h"

template<int width, int height>
class MTFrontierSearcher {
public:
	MTFrontierSearcher(
		SegmentedFile& frontierHoriz,
		SegmentedFile& frontierVert,
		SegmentedFile& new_frontierHoriz,
		SegmentedFile& new_frontierVert,
		SegmentedFile& exp,
		SegmentedFile& new_exp)
		: frontierReaderHoriz(frontierHoriz)
		, frontierReaderVert(frontierVert)
		, collector(new_frontierHoriz, new_frontierVert, new_exp)
		, r_exp(exp)
	{}

	void Collect(int segment) {
		r_exp.SetSegment(segment);
		frontierReaderHoriz.SetSegment(segment);
		frontierReaderVert.SetSegment(segment);
		collector.SetSegment(segment);

		bool empty = true;
		while (true) {
			auto& buf = r_exp.Read();
			if (buf.Size() == 0) break;
			empty = false;
			collector.AddCrossSegmentVerticalMoves(buf.Buf(), buf.Size());
		}
		while (true) {
			auto& buf = frontierReaderHoriz.Read();
			if (buf.IsEmpty()) break;
			empty = false;
			collector.AddHorizontalMoves(buf.Buf(), buf.Size());
			collector.AddExclude(buf.Buf(), buf.Size());
		}
		while (true) {
			auto& buf = frontierReaderVert.Read();
			if (buf.IsEmpty()) break;
			empty = false;
			collector.AddSameSegmentVerticalMoves(buf.Buf(), buf.Size());
			collector.AddExclude(buf.Buf(), buf.Size());
		}
		if (!empty) {
			auto cur = collector.SaveSegment();
			total += cur;
		}
	}

	void FinishCollect() {
		collector.CloseAll();
		total = 0;
	}

	uint64_t GetTotal() { return total; }

private:
	MTFrontierFileReader frontierReaderHoriz;
	MTFrontierFileReader frontierReaderVert;
	MTCollector<width, height> collector;
	ExpandedFrontierReader r_exp;

	uint64_t total = 0;
};

template<int width, int height>
std::vector<uint64_t> MTFrontierSearch(MTSearchOptions options) {
	std::cerr << "multi-tile metric, " << width << " x " << height << ", threads: " << options.Threads << '\n';
	std::cerr << "max depth: " << options.MaxDepth << "; ini: " << options.InitialValue << '\n';
	PrintVecor("FrontierVert1", options.FileFrontierVert1);
	PrintVecor("FrontierHoriz1", options.FileFrontierHoriz1);
	PrintVecor("FrontierVert2", options.FileFrontierVert2);
	PrintVecor("FrontierHoriz2", options.FileFrontierHoriz2);
	PrintVecor("Exp1", options.FileExpanded1);
	PrintVecor("Exp2", options.FileExpanded2);
	std::cerr << std::endl;

	Timer totalStart;
	Puzzle<width, height> puzzle;

	SegmentedFile frontierVert(puzzle.MaxSegments(), options.FileFrontierVert1);
	SegmentedFile frontierHoriz(puzzle.MaxSegments(), options.FileFrontierHoriz1);
	SegmentedFile new_frontierVert(puzzle.MaxSegments(), options.FileFrontierVert2);
	SegmentedFile new_frontierHoriz(puzzle.MaxSegments(), options.FileFrontierHoriz2);
	SegmentedFile e_exp(puzzle.MaxSegments(), options.FileExpanded1);
	SegmentedFile new_e_exp(puzzle.MaxSegments(), options.FileExpanded2);

	{
		auto [initialSegment, initialIndex] = puzzle.Rank(options.InitialValue);
		std::cerr << "Init: " << puzzle.Unrank(initialSegment, initialIndex) << std::endl;
		ensure(!puzzle.VerticalMoveChangesSegment(initialIndex % 16));
		// TODO: allow initial states with cross-segment vertical moves
		ensure((initialIndex % 16) % width == 0);
		MTFrontierFileWriter fwriterHoriz(frontierHoriz);
		fwriterHoriz.SetSegment(initialSegment);
		fwriterHoriz.Add(initialIndex);
		fwriterHoriz.FinishSegment();
		MTFrontierFileWriter fwriterVert(frontierVert);
		fwriterVert.SetSegment(initialSegment);
		fwriterVert.Add(initialIndex);
		fwriterVert.FinishSegment();
	}

	std::vector<std::unique_ptr<MTFrontierSearcher<width, height>>> searchers;
	for (int i = 0; i < options.Threads; i++) {
		searchers.emplace_back(
			std::make_unique<MTFrontierSearcher<width, height>>(
				frontierHoriz, frontierVert, new_frontierHoriz, new_frontierVert, e_exp, new_e_exp));
	}

	std::vector<uint64_t> widths;
	widths.push_back(1);

	std::cerr << "0: 1" << std::endl;

	while (widths.size() <= options.MaxDepth) {

		Timer timerStartStep;

		std::atomic<uint64_t> total = 0;

		std::atomic<int> segment = 0;
		auto fnCollect = [&](int index) {
			auto& searcher = *searchers[index];
			while (true) {
				int s = segment++;
				if (s >= puzzle.MaxSegments()) break;
				searcher.Collect(s);
			}
			total += searcher.GetTotal();
			searcher.FinishCollect();
		};

		std::vector<std::thread> threads;
		for (int i = 0; i < options.Threads; i++) {
			threads.emplace_back(fnCollect, i);
		}
		for (int i = 0; i < options.Threads; i++) {
			threads[i].join();
		}

		if (total == 0) break;

		std::cerr
			<< widths.size() << ": " << WithDecSep(total)
			<< " time=" << WithTime(timerStartStep.Elapsed())
			<< " Files=" << WithSize(frontierVert.TotalLength() + frontierHoriz.TotalLength())
			<< ", " << WithSize(e_exp.TotalLength())
			<< ", " << WithSize(new_frontierVert.TotalLength() + new_frontierHoriz.TotalLength())
			<< ", " << WithSize(new_e_exp.TotalLength())
			<< std::endl;

		widths.push_back(total);

		frontierVert.DeleteAll();
		frontierHoriz.DeleteAll();
		e_exp.DeleteAll();
		std::swap(frontierVert, new_frontierVert);
		std::swap(frontierHoriz, new_frontierHoriz);
		std::swap(e_exp, new_e_exp);
	}

	std::cerr << "Finished in " << WithTime(totalStart.Elapsed()) << std::endl;
	uint64_t totalStates = 0;
	for (auto w : widths) totalStates += w;
	std::cerr << "Total states: " << WithDecSep(totalStates) << std::endl;

	MTCollector<width, height>::PrintStats();
	GpuSolver<width, height>::PrintStats();
	SegmentedFile::PrintStats();
	StreamVInt::PrintStats();

	return widths;
}

template std::vector<uint64_t> MTFrontierSearch<2, 2>(MTSearchOptions options);
template std::vector<uint64_t> MTFrontierSearch<3, 2>(MTSearchOptions options);
template std::vector<uint64_t> MTFrontierSearch<4, 2>(MTSearchOptions options);
template std::vector<uint64_t> MTFrontierSearch<5, 2>(MTSearchOptions options);
template std::vector<uint64_t> MTFrontierSearch<6, 2>(MTSearchOptions options);
template std::vector<uint64_t> MTFrontierSearch<7, 2>(MTSearchOptions options);
template std::vector<uint64_t> MTFrontierSearch<8, 2>(MTSearchOptions options);

template std::vector<uint64_t> MTFrontierSearch<3, 3>(MTSearchOptions options);
template std::vector<uint64_t> MTFrontierSearch<4, 3>(MTSearchOptions options);
template std::vector<uint64_t> MTFrontierSearch<5, 3>(MTSearchOptions options);

template std::vector<uint64_t> MTFrontierSearch<4, 4>(MTSearchOptions options);
