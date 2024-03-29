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
    return Store::CreateFileStore(Segments, suffix, Opts.storeOptions);
}

StoreSet SegmentedOptions::MakeStoreSet(std::string suffix, int count) {
    StoreSet storeSet;
    for (int i = 0; i < count; i++) {
        storeSet.Stores.emplace_back(MakeStore(suffix + "-" + std::to_string(i)));
    }
    return storeSet;
}

void SegmentedOptions::PrintOptions() {
    std::cerr
        << Puzzle.Name()
        << "\nnodes: " << WithDecSep(TotalSize)
        << "; threads: " << Opts.threads
        << "\nseg.bits: " << Opts.segmentBits
        << "; segments: " << WithDecSep(Segments)
        << "; segment size: " << WithDecSep(SegmentSize)
        << std::endl;
}

void PrintResult(const std::vector<uint64_t>& result) {
    std::cerr << "Radius: " << result.size() - 1 << std::endl;
    std::cerr << "Result:";
    uint64_t sum = 0;
    for (auto w : result) {
        sum += w;
        std::cerr << " " << w;
    }
    std::cerr << "\nTotal: " << WithDecSep(sum) << std::endl;
}
