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
		SegmentedFile& frontierCS,
		SegmentedFile& new_frontierCS,
		SegmentedFile& e_up,
		SegmentedFile& e_dn)
		: frontier(frontier)
		, new_frontier(new_frontier)
		, frontierCS(frontierCS)
		, new_frontierCS(new_frontierCS)
		, e_up(e_up)
		, e_dn(e_dn)
		, frontierReader(frontier)
		, frontierReaderCS(frontierCS)
		, collector(new_frontier, new_frontierCS)
		, verticalCollector(e_up, e_dn)
		, r_up(e_up)
		, r_dn(e_dn)
	{}

	void Expand(int segment) {
		frontierReaderCS.SetSegment(segment);
		verticalCollector.SetSegment(segment);
		while (true) {
			auto read = frontierReaderCS.Read();
			if (read.Count == 0) break;
			verticalCollector.Add(read.Count, read.Indexes, read.Bounds);
		}
		verticalCollector.Close();
	}

	void Collect(int segment) {
		r_up.SetSegment(segment);
		r_dn.SetSegment(segment);
		frontierReader.SetSegment(segment);
		frontierReaderCS.SetSegment(segment);
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
		while (true) {
			auto buf = frontierReaderCS.Read();
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
		total = 0;
	}

	uint64_t GetTotal() { return total; }

private:
	SegmentedFile& frontier;
	SegmentedFile& new_frontier;
	SegmentedFile& frontierCS;
	SegmentedFile& new_frontierCS;
	SegmentedFile& e_up;
	SegmentedFile& e_dn;

	FrontierFileReader frontierReader;
	FrontierFileReader frontierReaderCS;
	Collector<width, height> collector;
	VerticalMovesCollector<width, height> verticalCollector;
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
	SegmentedFile frontierCS(puzzle.MaxSegments(), options.FileFrontierCS1);
	SegmentedFile new_frontierCS(puzzle.MaxSegments(), options.FileFrontierCS2);
	SegmentedFile e_up(puzzle.MaxSegments(), options.FileExpanded1);
	SegmentedFile e_dn(puzzle.MaxSegments(), options.FileExpanded2);

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
				frontier, new_frontier, frontierCS, new_frontierCS, e_up, e_dn));
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
		auto frontierSizeCS = frontierCS.TotalLength();

		{
			std::atomic<int> segment = 0;
			auto fnExpand = [&](int index) {
				while (true) {
					int s = segment++;
					if (s >= puzzle.MaxSegments()) break;
					searchers[index]->Expand(s);
				}
			};

			std::vector<std::thread> threads;
			for (int i = 0; i < options.Threads; i++) {
				threads.emplace_back(fnExpand, i);
			}
			for (int i = 0; i < options.Threads; i++) {
				threads[i].join();
			}
		}

		timer_stage_1 += timerStartStep.Elapsed();

		//stage 2

		auto expandedSize = e_up.TotalLength() + e_dn.TotalLength();

		Timer timerStartStage2;

		{
			std::atomic<int> segment = 0;
			auto fnCollect = [&](int index) {
				while (true) {
					int s = segment++;
					if (s >= puzzle.MaxSegments()) break;
					searchers[index]->Collect(s);
				}
			};

			std::vector<std::thread> threads;
			for (int i = 0; i < options.Threads; i++) {
				threads.emplace_back(fnCollect, i);
			}
			for (int i = 0; i < options.Threads; i++) {
				threads[i].join();
			}

		}

		uint64_t total = 0;
		for (auto& searcher : searchers) {
			total += searcher->GetTotal();
			searcher->FinishCollect();
		}
		if (total == 0) break;

		timer_stage_2 += timerStartStage2.Elapsed();

		auto newFrontierSize = new_frontier.TotalLength();
		auto newFrontierSizeCS = new_frontierCS.TotalLength();

		std::cerr
			<< widths.size() << ": " << WithDecSep(total)
			<< " time=" << WithTime(timerStartStep.Elapsed())
			<< " Files=" << WithSize(frontierSize) 
			<< ", " << WithSize(frontierSizeCS)
			<< ", " << WithSize(expandedSize)
			<< ", " << WithSize(newFrontierSize)
			<< ", " << WithSize(newFrontierSizeCS)
			<< std::endl;

		widths.push_back(total);
		frontier.DeleteAll();
		frontierCS.DeleteAll();
		std::swap(frontier, new_frontier);
		std::swap(frontierCS, new_frontierCS);
		e_up.DeleteAll();
		e_dn.DeleteAll();
	}
	 
	std::cerr << "Finished in " << WithTime(totalStart.Elapsed()) << std::endl;

	std::cerr << " stage 1: " << WithTime(timer_stage_1) << std::endl;
	std::cerr << " stage 2: " << WithTime(timer_stage_2) << std::endl;

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

