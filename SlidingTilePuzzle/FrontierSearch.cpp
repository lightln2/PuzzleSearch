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
	FrontierSearcher(SegmentedFile& frontier, SegmentedFile& new_frontier, SegmentedFile& e_up, SegmentedFile& e_dn)
		: frontier(frontier)
		, new_frontier(new_frontier)
		, e_up(e_up)
		, e_dn(e_dn)
		, frontierReader(frontier)
		, collector(new_frontier)
		, verticalCollector(e_up, e_dn)
		, r_up(e_up)
		, r_dn(e_dn)
	{}

	void Expand(int segment) {
		frontierReader.SetSegment(segment);
		verticalCollector.SetSegment(segment);
		while (true) {
			auto read = frontierReader.Read();
			if (read.Count == 0) break;
			verticalCollector.Add(read.Count, read.Indexes, read.Bounds);
		}
		verticalCollector.Close();
	}

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
		frontier.Delete(segment);
		e_up.Delete(segment);
		e_dn.Delete(segment);
	}

	void FinishCollect() {
		total = 0;
	}

	uint64_t GetTotal() { return total; }

private:
	SegmentedFile& frontier;
	SegmentedFile& new_frontier;
	SegmentedFile& e_up;
	SegmentedFile& e_dn;

	FrontierFileReader frontierReader;
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

	SegmentedFile frontier(puzzle.MaxSegments(), "c:/PUZ/frontier1");
	SegmentedFile new_frontier(puzzle.MaxSegments(), "c:/PUZ/frontier2");
	SegmentedFile e_up(puzzle.MaxSegments(), "c:/PUZ/expanded_up");
	SegmentedFile e_dn(puzzle.MaxSegments(), "c:/PUZ/expanded_dn");

	{
		auto [initialSegment, initialIndex] = puzzle.Rank(options.InitialValue);
		FrontierFileWriter fwriter(frontier);
		fwriter.SetSegment(initialSegment);
		fwriter.Add(initialIndex, puzzle.GetBounds(initialIndex));
		fwriter.FinishSegment();
	}

	FrontierSearcher<width, height> searcher1(frontier, new_frontier, e_up, e_dn);
	FrontierSearcher<width, height> searcher2(frontier, new_frontier, e_up, e_dn);

	uint64_t timer_stage_1 = 0;
	uint64_t timer_stage_2 = 0;

	std::vector<uint64_t> widths;
	widths.push_back(1);

	std::cerr << "0: 1" << std::endl;

	std::mutex lock;

	while (widths.size() <= options.MaxDepth) {

		// stage 1

		Timer timerStartStep;

		auto frontierSize = frontier.TotalLength();

		{
			int segment = 0;
			std::thread t1([&]() {
				while (true) {
					int s = -1;
					lock.lock();
					s = segment++;
					lock.unlock();
					if (s >= puzzle.MaxSegments()) break;
					searcher1.Expand(s);
				}
				});
			
			std::thread t2([&]() {
				while (true) {
					int s = -1;
					lock.lock();
					s = segment++;
					lock.unlock();
					if (s >= puzzle.MaxSegments()) break;
					searcher2.Expand(s);
				}
				});
			
			t1.join();
			t2.join();

		}
		//for (int segment = 0; segment < puzzle.MaxSegments(); segment++) {
		//	searcher.Expand(segment);
		//}

		timer_stage_1 += timerStartStep.Elapsed();

		//stage 2

		auto expandedSize = e_up.TotalLength() + e_dn.TotalLength();

		Timer timerStartStage2;

		{
			int segment = 0;
			std::thread t1([&]() {
				while (true) {
					int s = -1;
					lock.lock();
					s = segment++;
					lock.unlock();
					if (s >= puzzle.MaxSegments()) break;
					searcher1.Collect(s);
				}
				});
			
			std::thread t2([&]() {
				while (true) {
					int s = -1;
					lock.lock();
					s = segment++;
					lock.unlock();
					if (s >= puzzle.MaxSegments()) break;
					searcher2.Collect(s);
				}
				});
			
			t1.join();
			t2.join();
		}

		//for (int segment = 0; segment < puzzle.MaxSegments(); segment++) {
		//	searcher.Collect(segment);
		//}

		//uint64_t total = searcher.GetTotal();
		uint64_t total = searcher1.GetTotal() + searcher2.GetTotal();
		searcher1.FinishCollect();
		searcher2.FinishCollect();
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

