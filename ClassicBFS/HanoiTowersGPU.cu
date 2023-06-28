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


}

struct FPStateGPU {
    int next[32] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    int top[4] = { -1, -1, -1, -1 };
    int bottom[4] = { -1, -1, -1, -1 };

    __device__ void add(int peg, int disk) {
        if (top[peg] == -1) top[peg] = disk;
        else next[bottom[peg]] = disk;
        bottom[peg] = disk;
    }

    __device__ bool can_move(int srcPeg, int dstPeg) {
        if (srcPeg == dstPeg) return false;
        int srcDisk = top[srcPeg];
        if (srcDisk == -1) return false;
        int dstDisk = top[dstPeg];
        if (dstDisk != -1 && dstDisk < srcDisk) return false;
        return true;
    }

    __device__ bool move(int srcPeg, int dstPeg) {
        if (srcPeg == dstPeg) return false;
        int srcDisk = top[srcPeg];
        if (srcDisk == -1) return false;
        int dstDisk = top[dstPeg];
        if (dstDisk != -1 && dstDisk < srcDisk) return false;
        top[srcPeg] = next[srcDisk];
        if (top[srcPeg] == -1) bottom[srcPeg] = -1;
        if (dstDisk == -1) {
            top[dstPeg] = bottom[dstPeg] = srcDisk;
            next[srcDisk] = -1;
        }
        else {
            top[dstPeg] = srcDisk;
            next[srcDisk] = dstDisk;
        }
        return true;
    }

    __device__ int restore_symmetry(int dstPeg) {
        int pegs[4]{ 0, 1, 2, 3 };
        auto fn_restore = [&](int i, int j) {
            if (bottom[i] < bottom[j]) {
                gpu_swap(top[i], top[j]);
                gpu_swap(bottom[i], bottom[j]);
                gpu_swap(pegs[i], pegs[j]);
            }
        };
        fn_restore(2, 3);
        fn_restore(1, 2);
        fn_restore(2, 3);
        int invpegs[4];
        invpegs[pegs[0]] = 0;
        invpegs[pegs[1]] = 1;
        invpegs[pegs[2]] = 2;
        invpegs[pegs[3]] = 3;
        return invpegs[dstPeg];
    }
};

__device__ uint64_t StateToIndex(const FPStateGPU& state) {
    uint64_t index = 0;
    for (int peg = 0; peg < 4; peg++) {
        int top = state.top[peg];
        for (int disk = top; disk >= 0; disk = state.next[disk]) {
            index |= (uint64_t(peg) << (2 * disk));
        }
    }
    return index;
}

__device__ FPStateGPU IndexToState(int size, uint64_t index) {
    FPStateGPU state;
    for (int i = 0; i < 32; i++) state.next[i] = -1;
    for (int i = 0; i < 4; i++) state.top[i] = -1;
    for (int disk = 0; disk < size; disk++) {
        int peg = (index >> (2 * disk)) & 3;
        state.add(peg, disk);
    }
    return state;
}

__global__ void kernel_hanoi_towers_expand(uint64_t* indexes, uint64_t* expanded, int size, bool useSymmetry, uint64_t count) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    uint64_t index = indexes[i];
    int opBits = index & 15;
    index >>= 4;

    FPStateGPU state = IndexToState(size, index);
    uint64_t baseDst = i * 6;
    int pos = 0;
    for (int peg1 = 0; peg1 < 4; peg1++) {
        for (int peg2 = peg1 + 1; peg2 < 4; peg2++) {
            expanded[baseDst + pos] = INVALID_INDEX;
            FPStateGPU s2 = state;
            int p1 = peg1, p2 = peg2;
            if (!s2.can_move(p1, p2)) gpu_swap(p1, p2);
            if (s2.move(p1, p2)) {
                if (useSymmetry) {
                    p2 = s2.restore_symmetry(p2);
                }
                if (!HasOp(opBits, p1)) {
                    uint64_t childIndex = StateToIndex(s2);
                    expanded[baseDst + pos] = (childIndex << 4) | p2;
                }
            }
            pos++;
        }
    }
}

void GpuHanoiTowersExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int size,
    bool useSymmetry,
    uint64_t count,
    CuStream stream)
{
    auto threadsPerBlock = 256;
    auto blocksPerGrid = uint32_t((count + threadsPerBlock - 1) / threadsPerBlock);
    
    kernel_hanoi_towers_expand<<<blocksPerGrid, threadsPerBlock, 0, cudaStream_t(stream) >>> (
        gpuIndexes,
        gpuExpanded,
        size,
        useSymmetry,
        count);
    ERR(cudaGetLastError());
}


