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

template<int size>
struct FPStateGPU {
    int8_t next[size];
    int top[4];
    int bottom[4];

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

template<int size>
__device__ uint64_t StateToIndex(const FPStateGPU<size>& state) {
    uint64_t index = 0;
    for (int peg = 0; peg < 4; peg++) {
        int top = state.top[peg];
        for (int disk = top; disk >= 0; disk = state.next[disk]) {
            index |= (uint64_t(peg) << (2 * disk));
        }
    }
    return index;
}

template<int size>
__device__ FPStateGPU<size> IndexToState(uint64_t index) {
    FPStateGPU<size> state;
    for (int i = 0; i < size; i++) state.next[i] = -1;
    for (int i = 0; i < 4; i++) state.top[i] = -1;
    for (int i = 0; i < 4; i++) state.bottom[i] = -1;
    for (int disk = 0; disk < size; disk++) {
        int peg = (index >> (2 * disk)) & 3;
        state.add(peg, disk);
    }
    return state;
}

template<int size, bool useSymmetry>
__global__ void kernel_hanoi_towers_expand(uint64_t* indexes, uint64_t* expanded, uint64_t count) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    uint64_t index = indexes[i];
    int opBits = index & 15;
    index >>= 4;

    FPStateGPU<size> state = IndexToState<size>(index);
    uint64_t baseDst = i * 6;
    int pos = 0;
    for (int peg1 = 0; peg1 < 4; peg1++) {
        for (int peg2 = peg1 + 1; peg2 < 4; peg2++) {
            uint64_t result = INVALID_INDEX;
            FPStateGPU<size> s2 = state;
            int p1 = peg1, p2 = peg2;
            if (!s2.can_move(p1, p2)) gpu_swap(p1, p2);
            if (!HasOp(opBits, p1)) {
                if (s2.move(p1, p2)) {
                    if (useSymmetry) {
                        p2 = s2.restore_symmetry(p2);
                    }
                    uint64_t childIndex = StateToIndex(s2);
                    result = (childIndex << 4) | p2;
                }
            }
            expanded[baseDst + pos] = result;
            pos++;
        }
    }
}

template<int size, bool useSymmetry>
void GpuHanoiTowersExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    uint64_t count,
    CuStream stream)
{
    auto threadsPerBlock = 256;
    auto blocksPerGrid = uint32_t((count + threadsPerBlock - 1) / threadsPerBlock);
    
    kernel_hanoi_towers_expand<size, useSymmetry> << <blocksPerGrid, threadsPerBlock, 0, cudaStream_t(stream) >> > (
        gpuIndexes,
        gpuExpanded,
        count);
    ERR(cudaGetLastError());
}

template void GpuHanoiTowersExpand<10, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<11, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<12, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<13, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<14, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<15, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<16, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<17, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<18, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<19, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<20, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<21, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<22, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<23, false>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);

template void GpuHanoiTowersExpand<10, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<11, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<12, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<13, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<14, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<15, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<16, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<17, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<18, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<19, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<20, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<21, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<22, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
template void GpuHanoiTowersExpand<23, true>(uint64_t* gpuIndexes, uint64_t* gpuExpanded, uint64_t count, CuStream stream);
