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
    //return Store::CreateMultiFileStore(Segments, Opts.directories, suffix);
    return Store::CreateSingleFileStore(Segments, Opts.directories, suffix);
}

StoreSet SegmentedOptions::MakeStoreSet(std::string suffix, int count) {
    StoreSet storeSet;
    for (int i = 0; i < count; i++) {
        storeSet.Stores.emplace_back(MakeStore(suffix + "_" + std::to_string(i) + "_"));
    }
    return storeSet;
}

void SegmentedOptions::PrintOptions() {
    std::cerr
        << "; nodes: " << WithDecSep(TotalSize)
        << "; segments: " << WithDecSep(Segments)
        << "; segment size: " << WithDecSep(SegmentSize)
        << std::endl;
}
