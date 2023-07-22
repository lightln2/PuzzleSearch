#include "pch.h"

#include "../Common/Store.h"
#include "../Common/Util.h"
#include "../Common/SegmentReader.h"
#include "../Common/SegmentWriter.h"
#include "../Common/Multiplexor.h"

void TestStore(Store& store) {
	constexpr size_t SIZE = 10 * 1024;
	constexpr size_t BUFS = 300;
	int segments = store.MaxSegments();
	std::vector<uint8_t> buffer(SIZE);
	uint8_t* buf = &buffer[0];

	for (int i = 0; i < BUFS; i++) {
		memset(buf, i & 255, SIZE);
		int segment = i % segments;
		store.Write(segment, buf, SIZE);
	}

	for (int tries = 0; tries < 3; tries++) {
		for (int i = 0; i < BUFS; i++) {
			int segment = i % segments;
			size_t read = store.Read(segment, buf, SIZE);
			EXPECT_EQ(read, SIZE);
			for (int j = 0; j < SIZE; j++) ensure(buf[j] == (i & 255));
		}
		store.RewindAll();
	}

	store.DeleteAll();
	for (int i = 0; i < segments; i++) {
		size_t read = store.Read(i, buf, SIZE);
		EXPECT_EQ(read, 0);
	}

}

TEST(StoreTests, SingleFile) {
	auto store = Store::CreateSingleFileStore(50, "./file");
	TestStore(store);
}

TEST(StoreTests, MultiFile) {
	auto store = Store::CreateMultiFileStore(50, "./dir", "store");
	TestStore(store);
}

TEST(StoreTests, ParallelSingleFile) {
	StoreOptions opts;
	opts.directories = { "./dir1", "./dir2", "./dir3" };
	opts.filesPerPath = 1;
	auto store = Store::CreateFileStore(50, "store", opts);
	TestStore(store);
}

TEST(StoreTests, ParallelMultiFile) {
	StoreOptions opts;
	opts.directories = { "./dir1", "./dir2", "./dir3" };
	opts.filesPerPath = 0;
	auto store = Store::CreateFileStore(50, "store", opts);
	TestStore(store);
}

TEST(StoreTests, Sequential) {
	StoreOptions opts;
	opts.directories = { "./dir" };
	opts.filesPerPath = 3;
	auto store = Store::CreateFileStore(50, "store", opts);
	TestStore(store);
}

TEST(StoreTests, Generic) {
	StoreOptions opts;
	opts.directories = { "./dir1", "./dir2", "./dir3" };
	opts.filesPerPath = 7;
	auto store = Store::CreateFileStore(50, "store", opts);
	TestStore(store);
}

TEST(StoreTests, SwapStores) {
	auto store1 = Store::CreateFileStore(50, "store1", {});
	auto store2 = Store::CreateFileStore(50, "store1", {});
	std::swap(store1, store2);
}

TEST(StoreTests, TestReaderWriter) {
	constexpr int SEGMENTS = 6;
	constexpr uint32_t VALUES = 5 * 1024 * 1024;

	auto store = Store::CreateSequentialStore(SEGMENTS, { "./f1" });
	SegmentReader reader(store);
	SegmentWriter writer(store);

	for (int i = 0; i < SEGMENTS; i++) {
		writer.SetSegment(i);
		for (uint32_t j = 0; j < VALUES; j++) {
			writer.Add(j);
		}
		writer.Flush();
	}

	for (int i = 0; i < SEGMENTS; i++) {
		reader.SetSegment(i);
		uint64_t count = 0;
		uint64_t sum = 0;
		while (true) {
			auto& data = reader.Read();
			if (data.IsEmpty()) break;
			count += data.Size();
			for (size_t i = 0; i < data.Size(); i++) sum += data[i];
		}
		EXPECT_EQ(VALUES, count);
		EXPECT_EQ((uint64_t)VALUES * (VALUES - 1) / 2, sum);
	}
}

TEST(StoreTests, TestSimpleMultiplexor) {
	constexpr int SEGMENTS = 6;
	constexpr uint32_t VALUES = 5 * 1024 * 1024;

	auto store = Store::CreateSequentialStore(SEGMENTS, { "./f1" });
	SegmentReader reader(store);
	SimpleMultiplexor mp(store, SEGMENTS);

	for (uint32_t j = 0; j < VALUES; j++) {
		for (int i = 0; i < SEGMENTS; i++) {
			mp.Add(i, j);
		}
	}
	mp.FlushAllSegments();

	for (int i = 0; i < SEGMENTS; i++) {
		reader.SetSegment(i);
		uint64_t count = 0;
		uint64_t sum = 0;
		while (true) {
			auto& data = reader.Read();
			if (data.IsEmpty()) break;
			count += data.Size();
			for (size_t i = 0; i < data.Size(); i++) sum += data[i];
		}
		EXPECT_EQ(VALUES, count);
		EXPECT_EQ((uint64_t)VALUES * (VALUES - 1) / 2, sum);
	}
}

