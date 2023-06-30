#include "PuzzleGpu.h"
#include "GPU.h"

PuzzleGpu::Exec::Exec(int branchingFactor)
    : stream(CreateCudaStream())
    , gpuSrc(CreateGPUBuffer(MAX_INDEXES_BUFFER))
    , gpuDst(CreateGPUBuffer(MAX_INDEXES_BUFFER * branchingFactor))
{}

PuzzleGpu::Exec::~Exec()
{
    DestroyCudaStream(stream);
    DestroyGPUBuffer(gpuSrc);
    DestroyGPUBuffer(gpuDst);
}

PuzzleGpu::Exec* PuzzleGpu::AquireStream() {
    m_Mutex.lock();
    if (m_FreeStreams.empty()) {
        m_Streams.emplace_back(std::make_unique<Exec>(BranchingFactor()));
        m_FreeStreams.push_back(m_Streams.back().get());
    }
    auto* stream = m_FreeStreams.back();
    m_FreeStreams.pop_back();
    m_Mutex.unlock();
    return stream;
}

void PuzzleGpu::ReleaseStream(PuzzleGpu::Exec* stream) {
    m_Mutex.lock();
    m_FreeStreams.push_back(stream);
    m_Mutex.unlock();
}

void PuzzleGpu::CombineIndexAndOpBits(std::vector<uint64_t>& indexes, std::vector<int>& usedOperatorBits) {
    auto ops = OperatorsCount();
    for (uint64_t i = 0; i < indexes.size(); i++) {
        indexes[i] = (indexes[i] << ops) | usedOperatorBits[i];
    }
}

void PuzzleGpu::SplitIndexAndOps(std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators) {
    auto ops = OperatorsCount();
    auto opsMask = (1 << ops) - 1;
    for (int i = 0; i < expandedIndexes.size(); i++) {
        auto val = expandedIndexes[i];
        expandedOperators[i] = val & opsMask;
        // sign shift so that INVALID_INDEX remains the same
        expandedIndexes[i] = (uint64_t)((int64_t)val >> ops);
    }
}

void PuzzleGpu::Expand(
    std::vector<uint64_t>& indexes,
    std::vector<int>& usedOperatorBits,
    std::vector<uint64_t>& expandedIndexes,
    std::vector<int>& expandedOperators,
    ExpandHint hint)
{
    SetupOutputBuffers(expandedIndexes, expandedOperators);

    auto br = BranchingFactor();

    expandedIndexes.resize(indexes.size() * br);
    expandedOperators.resize(indexes.size() * br);

    CombineIndexAndOpBits(indexes, usedOperatorBits);

    auto* stream = AquireStream();

    CopyToGpu(&indexes[0], stream->gpuSrc, indexes.size(), stream->stream);

    ExpandGpu(stream->gpuSrc, stream->gpuDst, indexes.size(), stream->stream);

    CopyFromGpu(stream->gpuDst, &expandedIndexes[0], indexes.size() * br, stream->stream);

    ReleaseStream(stream);

    SplitIndexAndOps(expandedIndexes, expandedOperators);
}
