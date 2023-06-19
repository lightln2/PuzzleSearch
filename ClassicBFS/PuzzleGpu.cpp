#include "PuzzleGpu.h"
#include "GPU.h"

PuzzleGpu::Exec::Exec()
    : stream(CreateCudaStream())
    , gpuSrc(CreateGPUBuffer(MAX_INDEXES_BUFFER))
    , gpuDst(CreateGPUBuffer(MAX_INDEXES_BUFFER * 4))
{}

PuzzleGpu::Exec::~Exec()
{
    DestroyCudaStream(stream);
    DestroyGPUBuffer(gpuSrc);
    DestroyGPUBuffer(gpuDst);
}

void PuzzleGpu::Expand(
    std::vector<uint64_t>& indexes,
    std::vector<int>& usedOperatorBits,
    std::vector<uint64_t>& expandedIndexes,
    std::vector<int>& expandedOperators)
{
    auto ops = OperatorsCount();
    auto opsExp = 1ui64 << ops;
    if (expandedIndexes.capacity() < MAX_INDEXES_BUFFER * ops) {
        expandedIndexes.reserve(MAX_INDEXES_BUFFER * ops);
    }
    if (expandedOperators.capacity() < MAX_INDEXES_BUFFER * ops) {
        expandedOperators.reserve(MAX_INDEXES_BUFFER * ops);
    }
    expandedIndexes.clear();
    expandedOperators.clear();
    expandedIndexes.resize(indexes.size() * ops);
    expandedOperators.resize(indexes.size() * ops);

    for (uint64_t i = 0; i < indexes.size(); i++) {
        indexes[i] = (indexes[i] * opsExp) | usedOperatorBits[i];
    }

    m_Mutex.lock();
    if (m_FreeStreams.empty()) {
        m_Streams.emplace_back(std::make_unique<Exec>());
        m_FreeStreams.push_back(m_Streams.back().get());
        //std::cerr << "added new stream: " << m_Streams.size() << std::endl;
    }
    auto* stream = m_FreeStreams.back();
    m_FreeStreams.pop_back();
    m_Mutex.unlock();

    CopyToGpu(&indexes[0], stream->gpuSrc, indexes.size(), stream->stream);

    ExpandGpu(stream->gpuSrc, stream->gpuDst, indexes.size(), stream->stream);

    CopyFromGpu(stream->gpuDst, &expandedIndexes[0], indexes.size() * 4, stream->stream);

    m_Mutex.lock();
    m_FreeStreams.push_back(stream);
    m_Mutex.unlock();

    for (int i = 0; i < expandedIndexes.size(); i++) {
        auto val = expandedIndexes[i];
        expandedOperators[i] = val & 15;
        // sign shift so that INVALID_INDEX remains the same
        expandedIndexes[i] = (uint64_t)((int64_t)val >> 4);
    }
}
