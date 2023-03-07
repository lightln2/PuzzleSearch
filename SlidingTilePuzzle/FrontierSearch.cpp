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
#include "SegmentedFile.h"

void ClassicFrontierSearch() {
	auto START = clock();
	int MAX_SEGMENTS = 1000;
	int REAL_SEGMENTS = 2;
	file::DeleteDirectory("d:/temp/frontier1");
	SegmentedFile file(MAX_SEGMENTS, "d:/temp/frontier1");
	constexpr size_t SIZE = 1024 * 1024;
	std::vector<uint8_t> buffer(SIZE);
	uint8_t* buf = &buffer[0];
	memset(buf, 7, SIZE);

	for (int i = 0; i < MAX_SEGMENTS; i++) {
		file.Write(i % REAL_SEGMENTS, buf, SIZE);
	}

	auto FINISH = clock();
	std::cerr << "Written in " << WithDecSep(FINISH - START) << std::endl;
	std::cerr << "Total size: " << file.TotalLength() << std::endl;

	for (int i = 0; i < MAX_SEGMENTS; i++) {
		size_t read = file.Read(i % REAL_SEGMENTS, buf, SIZE);
		ensure(read == SIZE);
		//file.Delete(i);
	}
	size_t read = file.Read(7, buf, SIZE);
	ensure(read == 0);

	auto FINISH2 = clock();
	std::cerr << "Read in " << WithDecSep(FINISH2 - FINISH) << std::endl;

	file.DeleteAll();

	auto FINISH3 = clock();
	std::cerr << "Deleted in " << WithDecSep(FINISH3 - FINISH2) << std::endl;

}

