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
	FrontierFileWriter fwriter(file);
	FrontierFileReader freader(file);
	for (int segment = 0; segment < SEGMENTS; segment++) {
		fwriter.SetSegment(segment);
		for (int i = 0; i < COUNTS; i++) fwriter.Add(i, i & 15);
		fwriter.FinishSegment();
	}

	for (int segment = 0; segment < SEGMENTS; segment++) {
		freader.SetSegment(segment);
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
	ExpandedFrontierWriter fwriter(file);
	ExpandedFrontierReader freader(file);
	for (int segment = 0; segment < SEGMENTS; segment++) {
		fwriter.SetSegment(segment);
		for (int i = 0; i < COUNTS; i++) fwriter.Add(i);
		fwriter.FinishSegment();
	}

	for (int segment = 0; segment < SEGMENTS; segment++) {
		freader.SetSegment(segment);
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
	constexpr int COUNTS = 5 * 1000 * 1000;

	SegmentedFile file(SEGMENTS, "./testexpandedmultiplexor");
	Multiplexor mult(SEGMENTS, file);
	for (int i = 0; i < SEGMENTS * COUNTS; i++) {
		mult.Add(i % SEGMENTS, i);
	}
	mult.Close();

	ExpandedFrontierReader freader(file);
	for (int segment = 0; segment < SEGMENTS; segment++) {
		freader.SetSegment(segment);
		uint64_t total = 0;
		while (true) {
			auto& buf = freader.Read();
			if (buf.Size() == 0) break;
			for (int i = 0; i < buf.Size(); i++) {
				ENSURE_EQ(buf[i], segment + (total + i) * SEGMENTS);
			}
			total += buf.Size();
		}
		EXPECT_EQ(total, COUNTS);
	}

}

TEST(TestCollector, TestCollector) {
	constexpr int SEGMENTS = 3;
	constexpr int COUNTS = 1 * 1000 * 1000;

	SegmentedFile file(SEGMENTS, "./testcollector");
	Collector<4, 3> collector(file);
	collector.SetSegment(1);
	for (int i = 0; i < COUNTS; i++) {
		collector.Add(i, i & 15);
	}
	auto total = collector.SaveSegment();
	EXPECT_EQ(total, COUNTS / 16 * 15);

	FrontierFileReader freader(file);
	freader.SetSegment(1);
	int stotal = 0;
	while (true) {
		auto buf = freader.Read();
		if (buf.Count == 0) break;
		stotal += buf.Count;
		for (int i = 0; i < buf.Count; i++) {
			int blank = buf.Indexes[i] & 15;
			if (blank >= 12) continue;
			auto exp = blank | Puzzle<4, 3>::GetBounds(blank);
			ENSURE_EQ(exp, buf.Bounds[i]);
		}
	}
	EXPECT_EQ(stotal, COUNTS / 16 * 13);
}

TEST(TestCollector, TestCollectorWithSkips) {
	constexpr int SEGMENTS = 3;
	constexpr int COUNTS = 40 * 1000 * 1000;

	SegmentedFile file(SEGMENTS, "./testcollector");
	Collector<4, 3> collector(file);
	collector.SetSegment(1);
	for (int i = 0; i < COUNTS; i += 13) {
		collector.Add(i, i & 15);
	}
	auto total = collector.SaveSegment();
	EXPECT_EQ(total, 2884616);

	FrontierFileReader freader(file);
	freader.SetSegment(1);
	int stotal = 0;
	while (true) {
		auto buf = freader.Read();
		if (buf.Count == 0) break;
		stotal += buf.Count;
	}
	EXPECT_EQ(stotal, 2500000);
}


