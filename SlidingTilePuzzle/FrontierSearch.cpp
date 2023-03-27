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

#include "Collector.h"
#include "FrontierFile.h"
#include "FrontierSearch.h"
#include "Multiplexor.h"
#include "Puzzle.h"
#include "SegmentedFile.h"
#include "VerticalMovesCollector.h"

template<int width, int height>
class FrontierSearcher {
public:
	FrontierSearcher(
		SegmentedFile& frontier,
		SegmentedFile& new_frontier,
		SegmentedFile& e_up,
		SegmentedFile& e_dn,
		SegmentedFile& new_e_up,
		SegmentedFile& new_e_dn)
		: frontier(frontier)
		, new_frontier(new_frontier)
		, e_up(e_up)
		, e_dn(e_dn)
		, new_e_up(e_up)
		, new_e_dn(e_dn)
		, frontierReader(frontier)
		, collector(new_frontier, new_e_up, new_e_dn)
		, r_up(e_up)
		, r_dn(e_dn)
	{}

	void Collect(int segment) {
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
			collector.AddSameSegmentVerticalMoves(buf.Indexes, buf.Bounds, buf.Count);
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
	SegmentedFile& frontier;
	SegmentedFile& new_frontier;
	SegmentedFile& e_up;
	SegmentedFile& e_dn;
	SegmentedFile& new_e_up;
	SegmentedFile& new_e_dn;

	FrontierFileReader frontierReader;
	Collector<width, height> collector;
	ExpandedFrontierReader r_up;
	ExpandedFrontierReader r_dn;

	uint64_t total = 0;
};


template<int width, int height>
std::vector<uint64_t> FrontierSearch(SearchOptions options) {
	Timer totalStart;
	Puzzle<width, height> puzzle;

	SegmentedFile frontier(puzzle.MaxSegments(), options.FileFrontier1);
	SegmentedFile new_frontier(puzzle.MaxSegments(), options.FileFrontier2);
	SegmentedFile e_up(puzzle.MaxSegments(), options.FileExpandedUp1);
	SegmentedFile e_dn(puzzle.MaxSegments(), options.FileExpandedDown1);
	SegmentedFile new_e_up(puzzle.MaxSegments(), options.FileExpandedUp2);
	SegmentedFile new_e_dn(puzzle.MaxSegments(), options.FileExpandedDown2);

	{
		auto [initialSegment, initialIndex] = puzzle.Rank(options.InitialValue);
		ensure(!puzzle.VerticalMoveChangesSegment(initialIndex % 16));
		FrontierFileWriter fwriter(frontier);
		fwriter.SetSegment(initialSegment);
		fwriter.Add(initialIndex, puzzle.GetBounds(initialIndex));
		fwriter.FinishSegment();
	}

	std::vector<std::unique_ptr<FrontierSearcher<width, height>>> searchers;
	for (int i = 0; i < options.Threads; i++) {
		searchers.emplace_back(
			std::make_unique<FrontierSearcher<width, height>>(
				frontier, new_frontier, e_up, e_dn, new_e_up, new_e_dn));
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
			<< " Files=" << WithSize(frontier.TotalLength()) 
			<< ", " << WithSize(e_up.TotalLength() + e_dn.TotalLength())
			<< ", " << WithSize(new_frontier.TotalLength())
			<< ", " << WithSize(new_e_up.TotalLength() + new_e_dn.TotalLength())
			<< std::endl;

		widths.push_back(total);

		frontier.DeleteAll();
		e_up.DeleteAll();
		e_dn.DeleteAll();
		std::swap(frontier, new_frontier);
		std::swap(e_up, new_e_up);
		std::swap(e_dn, new_e_dn);
	}
	 
	std::cerr << "Finished in " << WithTime(totalStart.Elapsed()) << std::endl;

	Collector<width, height>::PrintStats();
	GpuSolver<width, height>::PrintStats();
	SegmentedFile::PrintStats();
	StreamVInt::PrintStats();

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

