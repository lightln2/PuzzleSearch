#include "pch.h"

#include "../SlidingTilePuzzle/Collector.h"
#include "../SlidingTilePuzzle/FrontierFile.h"
#include "../SlidingTilePuzzle/Multiplexor.h"
#include "../SlidingTilePuzzle/Puzzle.h"
#include "../SlidingTilePuzzle/SegmentedFile.h"
#include "../SlidingTilePuzzle/Util.h"

TEST(TestFrontierFile, FrontierFileReadWrite) {
	constexpr int SEGMENTS = 3;
	constexpr int COUNTS = 5 * 1000 * 1000;
	SegmentedFile file(100, "./testfrontierfile");
	FrontierFileWriter fwriter;
	FrontierFileReader freader;
	for (int segment = 0; segment < SEGMENTS; segment++) {
		fwriter.SetSegment(&file, segment);
		for (int i = 0; i < COUNTS; i++) fwriter.Add(i, i & 15);
		fwriter.FinishSegment();
	}

	for (int segment = 0; segment < SEGMENTS; segment++) {
		freader.SetSegment(&file, segment);
		size_t totalCount = 0;
		while (true) {
			auto buf = freader.Read();
			if (buf.Count == 0) break;
			for (int i = 0; i < buf.Count; i++) {
				ENSURE_EQ(buf.Indexes[i], totalCount + i);
				ENSURE_EQ(buf.Bounds[i], (totalCount + i) & 15);
			}
			totalCount += buf.Count;
		}
		EXPECT_EQ(totalCount, COUNTS);
	}

}

TEST(TestFrontierFile, ExpandedFrontierReadWrite) {
	constexpr int SEGMENTS = 3;
	constexpr int COUNTS = 500;
	SegmentedFile file(100, "./testexpanded");
	ExpandedFrontierWriter fwriter;
	ExpandedFrontierReader freader;
	for (int segment = 0; segment < SEGMENTS; segment++) {
		fwriter.SetSegment(&file, segment);
		for (int i = 0; i < COUNTS; i++) fwriter.Add(i);
		fwriter.FinishSegment();
	}

	for (int segment = 0; segment < SEGMENTS; segment++) {
		freader.SetSegment(&file, segment);
		size_t totalCount = 0;
		while (true) {
			auto& buf = freader.Read();
			if (buf.Size() == 0) break;
			for (int i = 0; i < buf.Size(); i++) {
				ENSURE_EQ(buf[i], totalCount + i);
			}
			totalCount += buf.Size();
		}
		EXPECT_EQ(totalCount, COUNTS);
	}

}

TEST(TestFrontierFile, TestMultiplexor) {
	constexpr int SEGMENTS = 3;
	constexpr int COUNTS = 1 * 1000 * 1000;

	SegmentedFile file(SEGMENTS, "./testexpandedmultiplexor");
	ExpandedFrontierWriter fwriter;
	Multiplexor mult(SEGMENTS, &file, fwriter);
	for (int i = 0; i < SEGMENTS * COUNTS; i++) {
		mult.Add(i % SEGMENTS, i);
	}
	mult.Close();

	ExpandedFrontierReader freader;
	for (int segment = 0; segment < SEGMENTS; segment++) {
		freader.SetSegment(&file, segment);
		auto& buf = freader.Read();
		EXPECT_EQ(buf.Size(), COUNTS);
		if (buf.Size() == 0) break;
		for (int i = 0; i < buf.Size(); i++) {
			ENSURE_EQ(buf[i], segment + i * SEGMENTS);
		}
	}

}

TEST(TestCollector, TestCollector) {
	constexpr int SEGMENTS = 3;
	constexpr int COUNTS = 1 * 1000 * 1000;

	Puzzle<4, 4> puzzle;
	SegmentedFile file(SEGMENTS, "./testcollector");
	FrontierFileWriter fwriter;
	Collector collector(Puzzle<4, 3>::MaxIndexesPerSegment(), fwriter);
	collector.SetSegment(1);
	fwriter.SetSegment(&file, 1);
	for (int i = 0; i < COUNTS; i++) {
		collector.Add(i, i & 15);
	}
	auto total = collector.SaveSegment();
	EXPECT_EQ(total, COUNTS / 16 * 15);

	FrontierFileReader freader;
	freader.SetSegment(&file, 1);
	auto buf = freader.Read();
	EXPECT_EQ(buf.Count, COUNTS / 16 * 14);
	for (int i = 0; i < buf.Count; i++) {
		ENSURE_EQ(buf.Indexes[i] & 15, buf.Bounds[i]);
	}
}

