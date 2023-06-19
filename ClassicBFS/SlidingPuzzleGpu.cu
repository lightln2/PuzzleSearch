#include "SlidingPuzzleGpu.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

constexpr int OP_UP = 0, OP_LEFT = 1, OP_RIGHT = 2, OP_DOWN = 3;
constexpr uint64_t INVALID_INDEX = std::numeric_limits<uint64_t>::max();

__device__ bool HasOp(int op, int dir) {
    return op & (1 << dir);
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