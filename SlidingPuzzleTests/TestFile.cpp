#include "pch.h"

#include "../SlidingTilePuzzle/File.h"
#include "../SlidingTilePuzzle/SegmentedFile.h"
#include "../SlidingTilePuzzle/Util.h"

#include <thread>
#include <vector>

TEST(TestFile, FileIO) {
	constexpr size_t SIZE = 10 * 1024;
	constexpr size_t BUFS = 300;
	std::vector<uint8_t> buffer(SIZE);
	uint8_t* buf = &buffer[0];

	file::CreateDirectory("./temp");
	file::FHANDLE fd = file::OpenFile("./temp/file1");

	for (int i = 0; i < BUFS; i++) {
		memset(buf, i & 255, SIZE);
		file::Write(fd, buf, SIZE);
	}

	file::Seek(fd, 0);

	for (int i = 0; i < BUFS; i++) {
		size_t read = file::Read(fd, buf, SIZE);
		EXPECT_EQ(read, SIZE);
		for (int j = 0; j < SIZE; j++) ensure(buf[j] == (i & 255));
	}
	size_t read = file::Read(fd, buf, SIZE);
	EXPECT_EQ(read, 0);
	file::CloseFile(fd);
}

TEST(TestFile, FileCreateDelete) {
	file::CreateDirectory("./temp");
	file::DeleteFile("./temp/file1");
	file::DeleteFile("./temp/file1");
	file::FHANDLE fd = file::OpenFile("./temp/file1");
	file::CloseFile(fd);
	fd = file::OpenFile("./temp/file1");
	file::CloseFile(fd);
}

TEST(TestFile, RWFile) {
	do {
		file::CreateDirectory("./temp");
		RWFile file("./temp/myfile");

		constexpr size_t SIZE = 10 * 1024;
		constexpr size_t BUFS = 300;
		std::vector<uint8_t> buffer(SIZE);
		uint8_t* buf = &buffer[0];

		for (int i = 0; i < BUFS; i++) {
			memset(buf, i & 255, SIZE);
			file.Write(buf, SIZE);
		}

		file.Rewind();

		for (int i = 0; i < BUFS; i++) {
			size_t read = file.Read(buf, SIZE);
			EXPECT_EQ(read, SIZE);
			for (int j = 0; j < SIZE; j++) ensure(buf[j] == (i & 255));
		}
		size_t read = file.Read(buf, SIZE);
		EXPECT_EQ(read, 0);
	} while (0);
}

TEST(TestFile, RWFileBuffer) {
	file::CreateDirectory("./temp");
	RWFile file("./temp/myfilebuf");
	constexpr size_t SIZE = 10 * 1024;
	constexpr size_t BUFS = 300;
	Buffer<uint32_t> buffer(SIZE);

	for (int i = 0; i < BUFS; i++) {
		memset(buffer.Buf(), i & 255, buffer.Capacity() * 4);
		buffer.SetSize(buffer.Capacity());
		file.Write(buffer);
	}

	file.Rewind();

	for (int i = 0; i < BUFS; i++) {
		file.Read(buffer);
		EXPECT_EQ(buffer.Size(), SIZE);
		for (int j = 0; j < SIZE; j++) ensure((buffer[j] & 255) == (i & 255));
	}
	file.Read(buffer);
	EXPECT_EQ(buffer.Size(), 0);
}

TEST(TestFile, SegmentedFile) {
	do {
		file::CreateDirectory("./temp");
		SegmentedFile myfile(2500, "./temp/frontier");

		constexpr size_t SIZE = 10 * 1024;
		constexpr size_t BUFS = 300;
		std::vector<uint8_t> buffer(SIZE);
		uint8_t* buf = &buffer[0];

		for (int i = 0; i < BUFS; i++) {
			memset(buf, i & 255, SIZE);
			int segment = i % 5;
			myfile.Write(segment, buf, SIZE);
		}

		for (int tries = 0; tries < 3; tries++) {
			myfile.RewindAll();

			for (int i = 0; i < BUFS; i++) {
				int segment = i % 5;
				size_t read = myfile.Read(segment, buf, SIZE);
				EXPECT_EQ(read, SIZE);
				for (int j = 0; j < SIZE; j++) ensure(buf[j] == (i & 255));
			}
		}
	} while (0);
}

TEST(TestFile, SegmentedFileDelete) {
	file::CreateDirectory("./temp");
	SegmentedFile myfile(2500, "./temp/frontier");

	constexpr size_t SIZE = 10 * 1024;
	constexpr size_t BUFS = 300;
	std::vector<uint8_t> buffer(SIZE);
	uint8_t* buf = &buffer[0];

	for (int i = 0; i < BUFS; i++) {
		memset(buf, i & 255, SIZE);
		int segment = i % 5;
		myfile.Write(segment, buf, SIZE);
	}
	EXPECT_EQ(myfile.Length(1), SIZE * BUFS / 5);
	EXPECT_EQ(myfile.Length(2), SIZE * BUFS / 5);

	myfile.RewindAll();
	myfile.DeleteAll();

	EXPECT_EQ(myfile.TotalLength(), 0);

	{
		auto read = myfile.Read(2, buf, SIZE);
		EXPECT_EQ(read, 0);
	}

	myfile.Write(77, buf, 1234);
	EXPECT_EQ(myfile.Read(77, buf, SIZE), 1234);
	EXPECT_EQ(myfile.Read(99, buf, SIZE), 0);
}

TEST(TestFile, Multithreaded) {
	file::CreateDirectory("./temp");
	SegmentedFile myfile(2500, "./temp/frontier");
	constexpr int BUFSIZE = 1 * 1024 * 1024;
	constexpr int TRIES = 20;
	constexpr int THREADS = 8;

	auto fnWrite = [&](int segment) {
		Buffer<uint8_t> buf(BUFSIZE);
		memset(buf.Buf(), segment, BUFSIZE);
		buf.SetSize(BUFSIZE);
		for (int i = 0; i < TRIES; i++) {
			myfile.Write(segment, buf);
		}
	};

	{
		std::vector<std::thread> threads;
		for (int i = 0; i < THREADS; i++) {
			threads.emplace_back(fnWrite, i);
		}
		for (int i = 0; i < THREADS; i++) {
			threads[i].join();
		}
	}

	myfile.RewindAll();

	auto fnRead = [&](int segment) {
		Buffer<uint8_t> buf(BUFSIZE);
		for (int i = 0; i < TRIES; i++) {
			myfile.Read(segment, buf);
			EXPECT_EQ(buf.Size(), BUFSIZE);
			for (int j = 0; j < BUFSIZE; j++) {
				ENSURE_EQ(buf[j], segment);
			}
		}
		myfile.Read(segment, buf);
		EXPECT_EQ(buf.Size(), 0);
	};

	{
		std::vector<std::thread> threads;
		for (int i = 0; i < THREADS; i++) {
			threads.emplace_back(fnRead, i);
		}
		for (int i = 0; i < THREADS; i++) {
			threads[i].join();
		}

	}
}

TEST(TestFile, MultithreadedSameSegment) {
	file::CreateDirectory("./temp");
	SegmentedFile myfile(2500, "./temp/frontier");
	constexpr int BUFSIZE = 1 * 1024 * 1024;
	constexpr int TRIES = 20;
	constexpr int THREADS = 8;
	constexpr int SEGMENT = 77;

	auto fnWrite = [&](int seed) {
		Buffer<uint8_t> buf(BUFSIZE);
		memset(buf.Buf(), seed, BUFSIZE);
		buf.SetSize(BUFSIZE);
		for (int i = 0; i < TRIES; i++) {
			myfile.Write(SEGMENT, buf);
		}
	};

	{
		std::vector<std::thread> threads;
		for (int i = 0; i < THREADS; i++) {
			threads.emplace_back(fnWrite, i);
		}
		for (int i = 0; i < THREADS; i++) {
			threads[i].join();
		}
	}

	myfile.RewindAll();

	int seeds[THREADS] = { 0 };

	Buffer<uint8_t> buf(BUFSIZE);
	while (true) {
		myfile.Read(SEGMENT, buf);
		if (buf.IsEmpty()) break;
		EXPECT_EQ(buf.Size(), BUFSIZE);
		int seed = buf[0];
		ensure(seed >= 0 && seed < THREADS);
		seeds[seed]++;
		for (int j = 0; j < BUFSIZE; j++) {
			ENSURE_EQ(buf[j], seed);
		}
	}
	for (int i = 0; i < THREADS; i++) {
		EXPECT_EQ(seeds[i], TRIES);
	}
}
