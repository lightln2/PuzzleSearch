#include "GpuSolver.h"
#include "Puzzle.h"
#include "Util.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <time.h>
#include <vector>
#include <fstream>

#include "FrontierSearch.h"
#include "File.h"

void ClassicFrontierSearch() {
	auto START = clock();
	file::DeleteFile("d:/temp/file1");
	auto fd = file::OpenFile("d:/temp/file1");
	constexpr size_t SIZE = 1024 * 1024;
	std::vector<uint8_t> buffer(SIZE);

	uint8_t* buf = &buffer[0];

	for (int i = 0; i < 100; i++) {
		memset(buf, i & 256, SIZE);
		file::Write(fd, buf, SIZE);
	}

	auto FINISH = clock();
	std::cerr << "Written in " << WithDecSep(FINISH - START) << std::endl;

	file::SeekBeginning(fd);
	for (int i = 0; i < 100; i++) {
		size_t read = file::Read(fd, buf, SIZE);
		ensure(read == SIZE);
	}
	size_t read = file::Read(fd, buf, SIZE);
	ensure(read == 0);

	auto FINISH2 = clock();
	std::cerr << "Read in " << WithDecSep(FINISH2 - FINISH) << std::endl;

	file::CloseFile(fd);


}

