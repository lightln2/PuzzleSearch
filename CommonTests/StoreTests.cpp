#include "pch.h"

#include "../Common/Store.h"
#include "../Common/Util.h"

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
	auto store = Store::CreateSingleFileStore(50, "./file0");
	TestStore(store);
}

TEST(StoreTests, MultiFile) {
	auto store = Store::CreateMultiFileStore(50, "./dir0");
	TestStore(store);
}

TEST(StoreTests, ParallelSingleFile) {
	auto store = Store::CreateSingleFileStore(50, { "./file1", "./file2", "./file3" });
	TestStore(store);
}

TEST(StoreTests, ParallelMultiFile) {
	auto store = Store::CreateMultiFileStore(50, { "./dir1", "./dir2", "./dir3" });
	TestStore(store);
}

TEST(StoreTests, SequentialMultiFile) {
	auto store = Store::CreateSequentialStore(50, { "./dir4", "./dir5", "./dir6" });
	TestStore(store);
}
