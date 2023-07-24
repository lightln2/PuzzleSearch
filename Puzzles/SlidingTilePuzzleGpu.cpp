#include "SlidingTilePuzzleGpu.h"
#include "SlidingPuzzleGpu.h"
#include "PermutationMap.h"
#include "../Common/Util.h"
#include "GPU.h"

#include <sstream>

SlidingTilePuzzleGpu::SlidingTilePuzzleGpu(int width, int height)
    : m_Width(width)
    , m_Height(height)
    , m_Size(width* height)
{
}

std::string SlidingTilePuzzleGpu::Name() const {
    std::ostringstream stream;
    stream
        << "Sliding-Tile GPU: " << m_Width << " x " << m_Height;
    return stream.str();
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

void SlidingTilePuzzleGpu::ExpandGpu(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    uint64_t count,
    CuStream stream)
{
    GpuSlidingTilePuzzleSimpleExpand(gpuIndexes, gpuExpanded, m_Width, m_Size, count, stream);
}
