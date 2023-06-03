#include "DiskBasedBFS.h"

SegmentedOptions::SegmentedOptions(class Puzzle& puzzle, PuzzleOptions& opts) 
    : Puzzle(puzzle)
    , Opts(opts)
{
    TotalSize = puzzle.IndexesCount();
    SegmentSize = 1ui64 << opts.segmentBits;
    SegmentMask = SegmentSize - 1;
    Segments = int((TotalSize + SegmentSize - 1) / SegmentSize);
    if (Segments == 1 && SegmentSize > TotalSize) {
        SegmentSize = TotalSize; // SegmentMask is still valid
    }
    OperatorsCount = puzzle.OperatorsCount();
    OperatorsMask = (1 << OperatorsCount) - 1;
    HasOddLengthCycles = puzzle.HasOddLengthCycles();
}

Store SegmentedOptions::MakeStore(std::string suffix) {
    return Store::CreateMultiFileStore(Segments, Opts.directories, suffix);
}

void SegmentedOptions::PrintOptions() {
    std::cerr << "DiskBasedClassicBFS"
        << "; nodes: " << WithDecSep(TotalSize)
        << "; segments: " << WithDecSep(Segments)
        << "; segment size: " << WithDecSep(SegmentSize)
        << std::endl;
}
