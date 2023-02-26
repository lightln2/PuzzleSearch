#include "../SlidingTilePuzzle/GpuSolver.h"

#include <iostream>
#include <memory>

int main() {
	GpuSolver<4, 4> gpuSolver;

	HostBuffer indexes, segments;

	for (uint32_t i = 0; i < indexes.SIZE; i++) {
		indexes.Buffer[i] = i * 16 + 5;
		segments.Buffer[i] = 0;
	}

	gpuSolver.GpuUp(indexes.Buffer, segments.Buffer, segments.SIZE);
	gpuSolver.GpuDown(indexes.Buffer, segments.Buffer, segments.SIZE);

	for (int i = 0; i < indexes.SIZE; i++) {
		if (indexes.Buffer[i] != i * 16 + 5) {
			std::cerr << "indexes error at " << i << ": exp " << i * 16 + 5 << "; but was " << indexes.Buffer[i] << std::endl;
			exit(0);
		}
		if (segments.Buffer[i] != 0) {
			std::cerr << "segments error at " << i << ": exp " << 0 << "; but was " << segments.Buffer[i] << std::endl;
			exit(0);
		}
	}
	return 0;
}