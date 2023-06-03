#include "SlidingTilePuzzleGpu.h"
#include "PermutationMap.h"
#include "../Common/Util.h"
#include "GPU.h"

#include <sstream>

namespace {
    const int OP_UP = 0, OP_LEFT = 1, OP_RIGHT = 2, OP_DOWN = 3;

    bool HasOp(int op, int dir) {
        return op & (1 << dir);
    }

    void Move(int* arr, int* newarr, int blank, int newblank) {
        memcpy(newarr, arr, 16 * sizeof(int));
        newarr[blank] = arr[newblank];
        newarr[newblank] = arr[blank];
    }

}

SlidingTilePuzzleGpu::Exec::Exec()
    : stream(CreateCudaStream())
    , gpuSrc(CreateGPUBuffer(MAX_INDEXES_BUFFER))
    , gpuDst(CreateGPUBuffer(MAX_INDEXES_BUFFER * 4))
{}

SlidingTilePuzzleGpu::Exec::~Exec()
{
    DestroyCudaStream(stream);
    DestroyGPUBuffer(gpuSrc);
    DestroyGPUBuffer(gpuDst);
}


SlidingTilePuzzleGpu::SlidingTilePuzzleGpu(int width, int height)
    : m_Width(width)
    , m_Height(height)
    , m_Size(width* height)
{
}

uint64_t SlidingTilePuzzleGpu::IndexesCount() const {
    uint64_t result = 1;
    for (int i = 1; i <= m_Size; i++) result *= i;
    return result;
}

std::string SlidingTilePuzzleGpu::ToString(uint64_t index) {
    std::ostringstream stream;
    int arr[16];
    PermutationUnrank(index, arr, m_Size);
    for (int i = 0; i < m_Size; i++) {
        if (i > 0) stream << ' ';
        stream << arr[i];
    }
    return stream.str();
}

uint64_t SlidingTilePuzzleGpu::Parse(std::string state) {
    std::istringstream stream(state);
    int arr[16];
    for (int i = 0; i < m_Size; i++) {
        stream >> arr[i];
    }
    return PermutationRank(arr, m_Size);
}

void SlidingTilePuzzleGpu::Expand(
    std::vector<uint64_t>& indexes,
    std::vector<int>& usedOperatorBits,
    std::vector<uint64_t>& expandedIndexes,
    std::vector<int>& expandedOperators)
{
    if (expandedIndexes.capacity() < MAX_INDEXES_BUFFER * 4) {
        expandedIndexes.reserve(MAX_INDEXES_BUFFER * 4);
    }
    if (expandedOperators.capacity() < MAX_INDEXES_BUFFER * 4) {
        expandedOperators.reserve(MAX_INDEXES_BUFFER * 4);
    }
    expandedIndexes.clear();
    expandedOperators.clear();
    expandedIndexes.resize(indexes.size() * 4);
    expandedOperators.resize(indexes.size() * 4);

    for (uint64_t i = 0; i < indexes.size(); i++) {
        indexes[i] = (indexes[i] * 16) | usedOperatorBits[i];
    }

    m_Mutex.lock();
    if (m_FreeStreams.empty()) {
        m_Streams.emplace_back(std::make_unique<Exec>());
        m_FreeStreams.push_back(m_Streams.back().get());
        std::cerr << "added new stream: " << m_Streams.size() << std::endl;
    }
    auto* stream = m_FreeStreams.back();
    m_FreeStreams.pop_back();
    m_Mutex.unlock();

    CopyToGpu(&indexes[0], stream->gpuSrc, indexes.size(), stream->stream);
    GpuSlidingTilePuzzleSimpleExpand(stream->gpuSrc, stream->gpuDst, m_Width, m_Size, indexes.size(), stream->stream);
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
