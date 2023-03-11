#include "pch.h"

#include "../SlidingTilePuzzle/File.h"
#include "../SlidingTilePuzzle/SegmentedFile.h"
#include "../SlidingTilePuzzle/Util.h"

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

	file::SeekBeginning(fd);

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

		myfile.RewindAll();

		for (int i = 0; i < BUFS; i++) {
			int segment = i % 5;
			size_t read = myfile.Read(segment, buf, SIZE);
			EXPECT_EQ(read, SIZE);
			for (int j = 0; j < SIZE; j++) ensure(buf[j] == (i & 255));
		}
	} while (0);
}

TEST(TestFile, SegmentedFileDelete) {
	do {
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

		myfile.Delete(2);
		EXPECT_EQ(myfile.Length(2), 0);

		{
			auto read = myfile.Read(2, buf, SIZE);
			EXPECT_EQ(read, 0);
		}

		myfile.Write(2, buf, SIZE);

		myfile.RewindAll();

		EXPECT_EQ(myfile.Length(2), SIZE);
		{
			auto read = myfile.Read(2, buf, SIZE);
			EXPECT_EQ(read, SIZE);
		}

		myfile.DeleteAll();
		EXPECT_EQ(myfile.TotalLength(), 0);
		{
			auto read = myfile.Read(2, buf, SIZE);
			EXPECT_EQ(read, 0);
		}
	} while (0);
}
