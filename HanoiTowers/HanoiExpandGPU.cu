#include "HanoiExpandGPU.h"


GpuBuffer::GpuBuffer()
    : Stream(CreateCudaStream())
    , Indexes(CreateGPUBuffer32(SRCBUFSIZE))
    , Children(CreateGPUBuffer32(DSTBUFSIZE))
{}

GpuBuffer::~GpuBuffer()
{
    DestroyCudaStream(Stream);
    DestroyGPUBuffer32(Indexes);
    DestroyGPUBuffer32(Children);
}

static ObjectPool<GpuBuffer> BufferPool;

// gpu impl

__device__ static void gpu_swap(int& x, int& y) {
    int temp = x;
    x = y;
    y = temp;
}

__device__ static void gpu_swap(uint64_t& x, uint64_t& y) {
    uint64_t temp = x;
    x = y;
    y = temp;
}

__device__ static int gpu_min(int x, int y) {
    return x < y ? x : y;
}

template<int size>
struct GpuFPState {
    uint64_t pegs[4] = { 0, 0, 0, 0 };
    int top[4] = { 255, 255, 255, 255 };

    __device__ void from_index(uint64_t index) {
        static constexpr uint64_t SIZE_MASK = (1ui64 << (2 * size)) - 1;
        static constexpr uint64_t BIT_MASK = 0x5555555555555555ui64 & SIZE_MASK;
        uint64_t p0 = index >> 1;
        uint64_t p1 = index;
        uint64_t p0i = ~p0;
        uint64_t p1i = ~p1;
        pegs[0] = (p0i & p1i) & BIT_MASK;
        pegs[1] = (p0i & p1) & BIT_MASK;
        pegs[2] = (p0 & p1i) & BIT_MASK;
        pegs[3] = (p0 & p1) & BIT_MASK;
        top[0] = (__ffsll(pegs[0]) - 1) & 255;
        top[1] = (__ffsll(pegs[1]) - 1) & 255;
        top[2] = (__ffsll(pegs[2]) - 1) & 255;
        top[3] = (__ffsll(pegs[3]) - 1) & 255;
    }

    __device__ uint64_t to_index() const {
        return pegs[1] | (pegs[2] << 1) | ((pegs[3] << 1) | pegs[3]);
    }

    __device__ bool empty(int peg) const {
        return pegs[peg] == 0;
    }

    __device__ void move(int srcPeg, int dstPeg) {
        int disk = gpu_min(top[srcPeg], top[dstPeg]);
        if (disk != 255) {
            uint64_t bit = 1ui64 << disk;
            pegs[srcPeg] ^= bit;
            pegs[dstPeg] ^= bit;
        }
    }

    __device__ void restore_symmetry() {
        int bottom[4];
        bottom[0] = 63 - __clzll(pegs[0]);
        bottom[1] = 63 - __clzll(pegs[1]);
        bottom[2] = 63 - __clzll(pegs[2]);
        bottom[3] = 63 - __clzll(pegs[3]);

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
    }

};


template<int size>
__global__ void kernel_expand_inseg(int segment, uint32_t* indexes, uint64_t count, uint32_t* children) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    uint64_t dstPos = i * 6;

    uint64_t segmentBase = uint64_t(segment) << 32;

    uint64_t index = segmentBase | indexes[i];

    GpuFPState<size> state;
    state.from_index(index);
    bool noMovesBreakSymmetry =
        __popcll(state.pegs[1]) >= 2 &&
        __popcll(state.pegs[2]) >= 2 &&
        __popcll(state.pegs[3]) >= 1;

    if (noMovesBreakSymmetry) {
        auto fnMove = [&](int peg1, int peg2) {
            GpuFPState<size> s2 = state;
            s2.move(peg1, peg2);
            uint64_t child = s2.to_index();
            if ((child >> 32) == segment) {
                children[dstPos++] = uint32_t(child);
            }
            else {
                children[dstPos++] = uint32_t(index);
            }
        };
        fnMove(0, 1);
        fnMove(0, 2);
        fnMove(0, 3);
        fnMove(1, 2);
        fnMove(1, 3);
        fnMove(2, 3);
    }
    else {
        auto fnMove = [&](int peg1, int peg2) {
            GpuFPState<size> s2 = state;
            bool srcEmpty = s2.empty(peg1) || s2.empty(peg2);
            s2.move(peg1, peg2);
            bool dstEmpty = s2.empty(peg1) || s2.empty(peg2);
            if (srcEmpty || dstEmpty) {
                s2.restore_symmetry();
            }
            uint64_t child = s2.to_index();
            if ((child >> 32) == segment) {
                children[dstPos++] = uint32_t(child);
            }
            else {
                children[dstPos++] = uint32_t(index);
            }
        };
        fnMove(0, 1);
        fnMove(0, 2);
        fnMove(0, 3);
        fnMove(1, 2);
        fnMove(1, 3);
        fnMove(2, 3);
    }
}

template<int size>
__global__ void kernel_expand_inseg_without_smallest(int segment, uint32_t* indexes, uint64_t count, uint32_t* children) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    uint64_t dstPos = i * 3;

    uint64_t segmentBase = uint64_t(segment) << 32;

    uint64_t index = segmentBase | indexes[i];

    int smallestDisk = index & 3;
    GpuFPState<size> state;
    state.from_index(index);
    bool noMovesBreakSymmetry =
        __popcll(state.pegs[1]) >= 2 &&
        __popcll(state.pegs[2]) >= 2 &&
        __popcll(state.pegs[3]) >= 1;

    static int disksWithoutSmallest[4][3] = {
        {1, 2, 3},
        {0, 2, 3},
        {0, 1, 3},
        {0, 1, 2}
    };

    if (noMovesBreakSymmetry) {
        auto fnMove = [&](int peg1, int peg2) {
            GpuFPState<size> s2 = state;
            s2.move(peg1, peg2);
            uint64_t child = s2.to_index();
            if ((child >> 32) == segment) {
                children[dstPos++] = uint32_t(child);
            }
            else {
                children[dstPos++] = uint32_t(index);
            }
        };

        auto& dws = disksWithoutSmallest[smallestDisk];
        fnMove(dws[0], dws[1]);
        fnMove(dws[0], dws[2]);
        fnMove(dws[1], dws[2]);
    }
    else {
        auto fnMove = [&](int peg1, int peg2) {
            GpuFPState<size> s2 = state;
            bool srcEmpty = s2.empty(peg1) || s2.empty(peg2);
            s2.move(peg1, peg2);
            bool dstEmpty = s2.empty(peg1) || s2.empty(peg2);
            if (srcEmpty || dstEmpty) {
                s2.restore_symmetry();
            }
            uint64_t child = s2.to_index();
            if ((child >> 32) == segment) {
                children[dstPos++] = uint32_t(child);
            }
            else {
                children[dstPos++] = uint32_t(index);
            }
        };
        auto& dws = disksWithoutSmallest[smallestDisk];
        fnMove(dws[0], dws[1]);
        fnMove(dws[0], dws[2]);
        fnMove(dws[1], dws[2]);
    }


}

template<int size>
void GpuExpandInSegment(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children) {
    size_t dstCount = count * 6;
    ensure(count <= GpuBuffer::SRCBUFSIZE);
    ensure(dstCount <= GpuBuffer::DSTBUFSIZE);
    auto* gpu = BufferPool.Aquire();
    CopyToGpu(indexes, gpu->Indexes, count, gpu->Stream);

    auto threadsPerBlock = 256;
    auto blocksPerGrid = uint32_t((count + threadsPerBlock - 1) / threadsPerBlock);
    kernel_expand_inseg<size> << <blocksPerGrid, threadsPerBlock, 0, cudaStream_t(gpu->Stream) >> > (
        segment,
        gpu->Indexes,
        count,
        gpu->Children);
    ERR(cudaGetLastError());

    children.resize(dstCount);
    CopyFromGpu(gpu->Children, &children[0], dstCount, gpu->Stream);
    BufferPool.Release(gpu);
}

template<int size>
void GpuExpandInSegmentWithoutSmallest(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children) {
    size_t dstCount = count * 3;
    ensure(count <= GpuBuffer::SRCBUFSIZE);
    ensure(dstCount <= GpuBuffer::DSTBUFSIZE);
    auto* gpu = BufferPool.Aquire();
    CopyToGpu(indexes, gpu->Indexes, count, gpu->Stream);

    auto threadsPerBlock = 256;
    auto blocksPerGrid = uint32_t((count + threadsPerBlock - 1) / threadsPerBlock);
    kernel_expand_inseg_without_smallest<size> << <blocksPerGrid, threadsPerBlock, 0, cudaStream_t(gpu->Stream) >> > (
        segment,
        gpu->Indexes,
        count,
        gpu->Children);
    ERR(cudaGetLastError());

    children.resize(dstCount);
    CopyFromGpu(gpu->Children, &children[0], dstCount, gpu->Stream);

    BufferPool.Release(gpu);
}

template void GpuExpandInSegment<14>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegment<15>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegment<16>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegment<17>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegment<18>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegment<19>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegment<20>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegment<21>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegment<22>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegment<23>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegment<24>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);

template void GpuExpandInSegmentWithoutSmallest<14>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegmentWithoutSmallest<15>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegmentWithoutSmallest<16>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegmentWithoutSmallest<17>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegmentWithoutSmallest<18>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegmentWithoutSmallest<19>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegmentWithoutSmallest<20>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegmentWithoutSmallest<21>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegmentWithoutSmallest<22>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegmentWithoutSmallest<23>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
template void GpuExpandInSegmentWithoutSmallest<24>(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);

