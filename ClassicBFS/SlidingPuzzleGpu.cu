#include "SlidingPuzzleGpu.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <stdio.h>

constexpr int OP_UP = 0, OP_LEFT = 1, OP_RIGHT = 2, OP_DOWN = 3;
constexpr uint64_t INVALID_INDEX = std::numeric_limits<uint64_t>::max();

namespace {

    __device__ bool HasOp(int op, int dir) {
        return op & (1 << dir);
    }

}
__device__ void Move(int* arr, int* newarr, int blank, int newblank) {
    for (int i = 0; i < 16; i++) newarr[i] = arr[i];
    newarr[blank] = arr[newblank];
    newarr[newblank] = arr[blank];
}

__global__ void kernel_sliding_tile_simple(uint64_t* indexes, uint64_t* expanded, int width, int size, uint64_t count) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    uint64_t index = indexes[i];
    int opBits = index & 15;
    index >>= 4;

    int arr[16];
    GpuPermutationUnrank(index, arr, size);

    int blank = -1;
    for (int i = 0; i < size; i++) {
        if (arr[i] == 0) {
            blank = i;
            //break;
        }
    }

    int newarr[16];

    if (blank >= width && !HasOp(opBits, OP_UP)) {
        Move(arr, newarr, blank, blank - width);
        uint64_t exp = GpuPermutationRank(newarr, size);
        expanded[i * 4 + 0] = (exp << 4) | OP_DOWN;
    }
    else {
        expanded[i * 4 + 0] = INVALID_INDEX;
    }

    if (blank < size - width && !HasOp(opBits, OP_DOWN)) {
        Move(arr, newarr, blank, blank + width);
        uint64_t exp = GpuPermutationRank(newarr, size);
        expanded[i * 4 + 1] = (exp << 4) | OP_UP;
    }
    else {
        expanded[i * 4 + 1] = INVALID_INDEX;
    }

    if (blank % width > 0 && !HasOp(opBits, OP_LEFT)) {
        Move(arr, newarr, blank, blank - 1);
        uint64_t exp = GpuPermutationRank(newarr, size);
        expanded[i * 4 + 2] = (exp << 4) | OP_RIGHT;
    }
    else {
        expanded[i * 4 + 2] = INVALID_INDEX;
    }

    if (blank % width < width - 1 && !HasOp(opBits, OP_RIGHT)) {
        Move(arr, newarr, blank, blank + 1);
        uint64_t exp = GpuPermutationRank(newarr, size);
        expanded[i * 4 + 3] = (exp << 4) | OP_LEFT;
    }
    else {
        expanded[i * 4 + 3] = INVALID_INDEX;
    }
}

void GpuSlidingTilePuzzleSimpleExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int width,
    int size,
    uint64_t count,
    CuStream stream)
{
    auto threadsPerBlock = 256;
    auto blocksPerGrid = uint32_t((count + threadsPerBlock - 1) / threadsPerBlock);
    
    kernel_sliding_tile_simple<<<blocksPerGrid, threadsPerBlock, 0, cudaStream_t(stream) >>> (
        gpuIndexes,
        gpuExpanded,
        width,
        size,
        count);
    ERR(cudaGetLastError());
}


/*
OPTIMIZED
*/

constexpr int OFFSET_ZERO = 15;

__device__ void from_segment(int* arr, uint32_t segment) {
    arr[12] = segment & 15;
    arr[13] = (segment >> 4) & 15;
    arr[14] = segment >> 8;
}

__device__ uint32_t to_segment(int* arr) {
    return arr[12] | (arr[13] << 4) | (arr[14] << 8);
}

template <int size>
__device__ void from_index(int* arr, uint32_t index) {
    arr[OFFSET_ZERO] = index % 16;
    index /= 16;

    arr[0] = 0;
    arr[1] = 0;

    for (uint32_t i = 2; i < 12; i++) {
        uint32_t div = i + 1;
        uint32_t newIndex = index / div;
        arr[i] = index - newIndex * div;
        index = newIndex;
    }
}

template <int size>
__device__ uint32_t to_index(int* arr) {
    auto blank = arr[OFFSET_ZERO];
    uint32_t index = 0;
    for (uint32_t i = 11; i >= 2; i--)
    {
        uint32_t div = i + 1;
        index = index * div + arr[i];
    }
    return index * 16 + blank;
}

template <int size>
__device__ void pack(int* arr) {
    arr[0] = 0;
    arr[1] = 0;
    for (int i = size - 2; i > 2; i--) {
        for (int j = 2; j < i; j++) {
            arr[j] -= (int)(arr[j] >= arr[i]);
        }
    }
}

template <int size, int width, bool widthIsEven>
__device__ void unpack(int* arr) {
    bool invEven = true;
    for (int i = 1; i < size - 1; i++) {
        for (int j = 0; j < i; j++) {
            bool inv = (arr[j] >= arr[i]);
            arr[j] += (int)inv;
            invEven ^= !inv;
        }
    }
    // restore parity
    bool rowEven = ((arr[OFFSET_ZERO] / width) & 1) == 0;
    bool restoreParity = (widthIsEven && !invEven == rowEven) || (!widthIsEven && !invEven);
    if (restoreParity) {
        auto tmp = arr[0];
        arr[0] = arr[1];
        arr[1] = tmp;
    }
}

template <int width>
__device__ bool CanRotateUp(int* arr) {
    int zeroPos = arr[OFFSET_ZERO];
    return zeroPos >= width;
}

template <int size, int width>
__device__ bool CanRotateDn(int* arr) {
    int zeroPos = arr[OFFSET_ZERO];
    return zeroPos < size - width;
}

template <int width>
__device__ void RotateUp(int* arr) {
    int zeroPos = arr[OFFSET_ZERO];
    char cur = arr[zeroPos - width];
    for (int i = zeroPos - width; i < zeroPos - 1; i++) arr[i] = arr[i + 1];
    arr[zeroPos - 1] = cur;
    arr[OFFSET_ZERO] -= width;
}

template <int width>
__device__ void RotateDn(int* arr) {
    int zeroPos = arr[OFFSET_ZERO];
    char cur = arr[zeroPos + width - 1];
    for (int i = zeroPos + width - 1; i > zeroPos; i--) arr[i] = arr[i - 1];
    arr[zeroPos] = cur;
    arr[OFFSET_ZERO] += width;
}


template<int width, int height>
__global__ void kernel_sliding_tile_optimized(uint64_t* indexes, uint64_t* expanded, uint64_t count) {
    constexpr int size = width * height;

    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    uint64_t index = indexes[i];
    int opBits = index & 15;
    index >>= 4;

    int arr[16];

    uint32_t seg = uint32_t(index >> 32);
    uint32_t idx = uint32_t(index & 0xFFFFFFFF);
    from_segment(arr, seg);
    from_index<size>(arr, idx);
    unpack<size, width, width % 2 == 0>(arr);
    int blank = arr[OFFSET_ZERO];

    if (blank >= width && !HasOp(opBits, OP_UP)) {
        int arr2[16];
        for (int j = 0; j < 16; j++) arr2[j] = arr[j];
        RotateUp<width>(arr2);
        pack<size>(arr2);
        uint64_t newseg = to_segment(arr2);
        uint64_t newidx = to_index<size>(arr2);
        uint64_t exp = (newseg << 32) | newidx;
        expanded[i * 4 + 0] = (exp << 4) | OP_DOWN;
    }
    else {
        expanded[i * 4 + 0] = INVALID_INDEX;
    }

    if (blank < size - width && !HasOp(opBits, OP_DOWN)) {
        int arr2[16];
        for (int j = 0; j < 16; j++) arr2[j] = arr[j];
        RotateDn<width>(arr2);
        pack<size>(arr2);
        uint64_t newseg = to_segment(arr2);
        uint64_t newidx = to_index<size>(arr2);
        uint64_t exp = (newseg << 32) | newidx;
        expanded[i * 4 + 1] = (exp << 4) | OP_UP;
    }
    else {
        expanded[i * 4 + 1] = INVALID_INDEX;
    }

    if (blank % width > 0 && !HasOp(opBits, OP_LEFT)) {
        uint64_t exp = index - 1;
        expanded[i * 4 + 2] = (exp << 4) | OP_RIGHT;
    }
    else {
        expanded[i * 4 + 2] = INVALID_INDEX;
    }

    if (blank % width < width - 1 && !HasOp(opBits, OP_RIGHT)) {
        uint64_t exp = index + 1;
        expanded[i * 4 + 3] = (exp << 4) | OP_LEFT;
    }
    else {
        expanded[i * 4 + 3] = INVALID_INDEX;
    }
}

template<int width, int height>
void GpuSlidingTilePuzzleOptimizedExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    uint64_t count,
    CuStream stream)
{
    auto threadsPerBlock = 256;
    auto blocksPerGrid = uint32_t((count + threadsPerBlock - 1) / threadsPerBlock);

    kernel_sliding_tile_optimized<width, height> << <blocksPerGrid, threadsPerBlock, 0, cudaStream_t(stream) >> > (
        gpuIndexes,
        gpuExpanded,
        count);
    ERR(cudaGetLastError());
}

template
void GpuSlidingTilePuzzleOptimizedExpand<2, 2>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template
void GpuSlidingTilePuzzleOptimizedExpand<3, 2>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template
void GpuSlidingTilePuzzleOptimizedExpand<3, 3>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template
void GpuSlidingTilePuzzleOptimizedExpand<4, 2>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template
void GpuSlidingTilePuzzleOptimizedExpand<4, 3>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template
void GpuSlidingTilePuzzleOptimizedExpand<4, 4>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template
void GpuSlidingTilePuzzleOptimizedExpand<5, 2>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template
void GpuSlidingTilePuzzleOptimizedExpand<5, 3>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template
void GpuSlidingTilePuzzleOptimizedExpand<6, 2>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template
void GpuSlidingTilePuzzleOptimizedExpand<7, 2>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template
void GpuSlidingTilePuzzleOptimizedExpand<8, 2>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
