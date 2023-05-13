#include "pch.h"

#include "../Common/File.h"

#include <vector>

TEST(FileTests, FileIO) {
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
