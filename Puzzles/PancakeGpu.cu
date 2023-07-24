#include "SlidingPuzzleGpu.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <stdio.h>

constexpr uint64_t INVALID_INDEX = std::numeric_limits<uint64_t>::max();

namespace {

    __device__ bool HasOp(int op, int dir) {
        return op & (1 << dir);
    }

    __device__ void gpu_swap(int& x, int& y) {
        int temp = x;
        x = y;
        y = temp;
    }

    __device__ void MoveInternal(int* newarr, int size, int count) {
        int start = size - count, end = size - 1;
        while (start < end) {
            gpu_swap(newarr[start], newarr[end]);
            start++;
            end--;
        }
    }

    __device__ void Move(int* arr, int* newarr, int size, int count) {
        for (int i = 0; i < 16; i++) newarr[i] = arr[i];
        MoveInternal(newarr, size, count);
    }

    __device__ uint64_t OptPermutationRank(int* arr, int size) {
        GpuPermutationCompact(arr, size);
        if (size <= 12) {
            uint64_t index = 0;
            for (int i = 0; i < size; i++) {
                index *= (size - i);
                index += arr[i];
            }

            return index;
        }
        else {
            uint64_t index = 0;
            for (int i = size - 12; i < size; i++) {
                index *= (size - i);
                index += arr[i];
            }

            uint64_t segment = 0;
            for (int i = 0; i < size - 12; i++) {
                segment *= (size - i);
                segment += arr[i];
            }

            return (segment << 29) | index;
        }
    }

    __device__ void OptPermutationUnrank(uint64_t index, int* arr, int size) {
        if (size <= 12) {
            for (int i = size - 1; i >= 0; i--) {
                arr[i] = index % (size - i);
                index /= (size - i);
            }
        }
        else {
            uint64_t segment = index >> 29;
            index &= ((1ui64 << 29) - 1);

            for (int i = size - 13; i >= 0; i--) {
                arr[i] = segment % (size - i);
                segment /= (size - i);
            }

            for (int i = size - 1; i >= size - 12; i--) {
                arr[i] = index % (size - i);
                index /= (size - i);
            }
        }

        GpuPermutationUncompact(arr, size);
    }

}

__global__ void kernel_pancake_expand(uint64_t* indexes, uint64_t* expanded, int size, uint64_t count) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const int MAX_OP = size - 1;
    const uint64_t OP_MASK = (1ui64 << MAX_OP) - 1;

    uint64_t index = indexes[i];
    int opBits = index & OP_MASK;
    index >>= MAX_OP;

    int arr[16];
    OptPermutationUnrank(index, arr, size);

    int newarr[16];

    uint64_t dstBase = i * MAX_OP;
    for (int op = 0; op < MAX_OP; op++) {
        uint64_t result = INVALID_INDEX;
        if (!HasOp(opBits, op)) {
            Move(arr, newarr, size, op + 2);
            uint64_t child = OptPermutationRank(newarr, size);
            result = (child << MAX_OP) | op;
        }
        expanded[dstBase + op] = result;
    }
}

__global__ void kernel_pancake_expandInSegment(uint64_t* indexes, uint64_t* expanded, int size, uint64_t count) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const int MAX_OP = size - 1;
    const uint64_t OP_MASK = (1ui64 << MAX_OP) - 1;

    uint64_t index = indexes[i];
    int opBits = index & OP_MASK;
    index >>= MAX_OP;

    int arr[16];
    OptPermutationUnrank(index, arr, size);

    int newarr[16];

    uint64_t dstBase = i * 11;
    for (int op = 0; op < 11; op++) {
        uint64_t result = INVALID_INDEX;
        if (!HasOp(opBits, op)) {
            Move(arr, newarr, size, op + 2);
            uint64_t child = OptPermutationRank(newarr, size);
            result = (child << MAX_OP) | op;
        }
        expanded[dstBase + op] = result;
    }
}

__global__ void kernel_pancake_expandCrossSegment(uint64_t* indexes, uint64_t* expanded, int size, bool invIndex, uint64_t count) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const int MAX_OP = size - 1;
    const uint64_t OP_MASK = (1ui64 << MAX_OP) - 1;

    uint64_t index = indexes[i];
    int opBits = index & OP_MASK;
    index >>= MAX_OP;

    int arr[16];
    OptPermutationUnrank(index, arr, size);

    int newarr[16];

    uint64_t dstBase = i * (MAX_OP - 11) - 11;
    for (int op = 11; op < MAX_OP; op++) {
        uint64_t result = INVALID_INDEX;
        if (!HasOp(opBits, op)) {
            Move(arr, newarr, size, op + 2);
            if (size > 12 && invIndex) {
                MoveInternal(newarr, size, 12);
            }
            uint64_t child = OptPermutationRank(newarr, size);
            result = (child << MAX_OP) | op;
        }
        expanded[dstBase + op] = result;
    }
}

__global__ void kernel_pancake_crossSegmentPostProcess(uint32_t* indexes, int segment, int size, uint64_t count) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const uint64_t SEG_MASK = (1ui64 << 29) - 1;

    uint64_t index = (uint64_t(segment) << 29) | indexes[i];

    int arr[16];
    OptPermutationUnrank(index, arr, size);
    MoveInternal(arr, size, 12);
    index = OptPermutationRank(arr, size);
    if (segment != index >> 29) index = INVALID_INDEX;
    indexes[i] = uint32_t(index & SEG_MASK);
}

void PancakeExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int size,
    uint64_t count,
    CuStream stream)
{
    auto threadsPerBlock = 256;
    auto blocksPerGrid = uint32_t((count + threadsPerBlock - 1) / threadsPerBlock);
    
    kernel_pancake_expand<<<blocksPerGrid, threadsPerBlock, 0, cudaStream_t(stream) >>> (
        gpuIndexes,
        gpuExpanded,
        size,
        count);
    ERR(cudaGetLastError());
}

void PancakeExpandInSegment(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int size,
    uint64_t count,
    CuStream stream)
{
    auto threadsPerBlock = 256;
    auto blocksPerGrid = uint32_t((count + threadsPerBlock - 1) / threadsPerBlock);

    kernel_pancake_expandInSegment << <blocksPerGrid, threadsPerBlock, 0, cudaStream_t(stream) >> > (
        gpuIndexes,
        gpuExpanded,
        size,
        count);
    ERR(cudaGetLastError());
}

void PancakeExpandCrossSegment(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int size,
    bool invIndex,
    uint64_t count,
    CuStream stream)
{
    auto threadsPerBlock = 256;
    auto blocksPerGrid = uint32_t((count + threadsPerBlock - 1) / threadsPerBlock);

    kernel_pancake_expandCrossSegment << <blocksPerGrid, threadsPerBlock, 0, cudaStream_t(stream) >> > (
        gpuIndexes,
        gpuExpanded,
        size,
        invIndex,
        count);
    ERR(cudaGetLastError());
}

void PancakeCrossSegmentPostProcessGPU(
    uint32_t* gpuIndexes,
    int segment,
    int size,
    uint64_t count,
    CuStream stream)
{
    auto threadsPerBlock = 256;
    auto blocksPerGrid = uint32_t((count + threadsPerBlock - 1) / threadsPerBlock);

    kernel_pancake_crossSegmentPostProcess << <blocksPerGrid, threadsPerBlock, 0, cudaStream_t(stream) >> > (
        gpuIndexes,
        segment,
        size,
        count);
    ERR(cudaGetLastError());
}
