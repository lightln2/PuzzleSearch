#include "GpuSolver.h"
#include "gpu.h"
#include "Util.h"

#include <sstream>
#include <stdio.h>

constexpr size_t OFFSET_ZERO = 15;

__device__ __forceinline__ void from_segment(int* arr, uint32_t segment) {
    arr[12] = segment & 15;
    arr[13] = (segment >> 4) & 15;
    arr[14] = segment >> 8;
}

__device__ __forceinline__ uint32_t to_segment(int* arr) {
    return arr[12] | (arr[13] << 4) | (arr[14] << 8);
}

template <int size>
__device__ __forceinline__ void from_index(int* arr, uint32_t index) {
    arr[OFFSET_ZERO] = index % 16;
    index /= 16;

    arr[0] = 0;
    arr[1] = 0;

    #pragma unroll
    for (uint32_t i = 2; i < 12; i++) {
        uint32_t div = i + 1;
        uint32_t newIndex = index / div;
        arr[i] = index - newIndex * div;
        index = newIndex;
    }
}

template <int size>
__device__ __forceinline__ int32_t to_index(int* arr) {
    auto blank = arr[OFFSET_ZERO];
    uint32_t index = 0;
    #pragma unroll
    for (uint32_t i = 11; i >= 2; i--)
    {
        uint32_t div = i + 1;
        index = index * div + arr[i];
    }
    return index * 16 + blank;
}

template <int size>
__device__ __forceinline__ void pack(int* arr) {
    arr[0] = 0;
    arr[1] = 0;

    #pragma unroll
    for (int i = 2; i < size - 2; i++) {
        int x = arr[i];
        #pragma unroll
        for (int j = i + 1; j < size - 1; j++) {
            x -= (int)(arr[i] >= arr[j]);
        }
        arr[i] = x;
    }
    /*
    #pragma unroll
    for (int i = size - 2; i > 2; i--) {
        #pragma unroll
        for (int j = 2; j < i; j++) {
            arr[j] -= (int)(arr[j] >= arr[i]);
        }
    }
    */
}

template <int size, int width, bool widthIsEven>
__device__ __forceinline__ void unpack(int* arr) {
    bool invEven = true;

    #pragma unroll
    for (int i = 1; i < size - 1; i++) {
        #pragma unroll
        for (int j = 0; j < i; j++) {
            bool inv = (arr[j] >= arr[i]);
            arr[j] += (int)inv;
            invEven ^= !inv;
        }
    }

    // restore parity
    bool rowEven = ((arr[OFFSET_ZERO] / width) & 1) == 0;
    bool restoreParity = (widthIsEven && invEven == rowEven) || (!widthIsEven && invEven);
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
__device__ __forceinline__ void RotateUp(int* arr) {
    int zeroPos = arr[OFFSET_ZERO];
    char cur = arr[zeroPos - width];
    for (int i = zeroPos - width; i < zeroPos - 1; i++) arr[i] = arr[i + 1];
    arr[zeroPos - 1] = cur;
    arr[OFFSET_ZERO] -= width;
}

template <int width>
__device__ __forceinline__ void RotateDn(int* arr) {
    int zeroPos = arr[OFFSET_ZERO];
    char cur = arr[zeroPos + width - 1];
    for (int i = zeroPos + width - 1; i > zeroPos; i--) arr[i] = arr[i - 1];
    arr[zeroPos] = cur;
    arr[OFFSET_ZERO] += width;
}

template<int width, int height>
__global__ void kernel_up(uint32_t segment, uint32_t* indexes, uint32_t* out_segments, size_t count) {
    constexpr int size = width * height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int arr[16];
    from_segment(arr, segment);
    from_index<size>(arr, indexes[i]);
    unpack<size, width, width % 2 == 0>(arr);
    /*
#ifdef _DEBUG
    if (!CanRotateUp<width>(arr)) {
        indexes[i] = (uint32_t)-1;
        out_segments[i] = (uint32_t)-1;
        return;
    }
#endif
    */
    RotateUp<width>(arr);
    pack<size>(arr);
    out_segments[i] = to_segment(arr);
    indexes[i] = to_index<size>(arr);
}

template<int width, int height>
__global__ void kernel_dn(uint32_t segment, uint32_t* indexes, uint32_t* out_segments, size_t count) {
    constexpr int size = width * height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int arr[16];
    from_segment(arr, segment);
    from_index<size>(arr, indexes[i]);
    unpack<size, width, width % 2 == 0>(arr);
    /*
#ifdef _DEBUG
    if (!CanRotateDn<size, width>(arr)) {
        indexes[i] = (uint32_t)-1;
        out_segments[i] = (uint32_t)-1;
        return;
    }
#endif
    */
    RotateDn<width>(arr);
    pack<size>(arr);
    out_segments[i] = to_segment(arr);
    indexes[i] = to_index<size>(arr);
}

HostBuffer::HostBuffer() {
    ERR(cudaHostAlloc(&Buffer, GPU_BUFFER_SIZE * sizeof(int32_t), cudaHostAllocDefault));
}

HostBuffer::~HostBuffer() {
    ERR(cudaFreeHost(Buffer));
}

template<int width, int height>
std::atomic<uint64_t> GpuSolver<width, height>::StatProcessedStates = 0;
template<int width, int height>
std::atomic<uint64_t> GpuSolver<width, height>::StatExecutionNanos = 0;

template<int width, int height>
GpuSolver<width, height>::GpuSolver() {
    ERR(cudaMalloc((void**)&GpuIndexesBuffer, GPU_BUFFER_SIZE * sizeof(int32_t)));
    ERR(cudaMalloc((void**)&GpuSegmentsBuffer, GPU_BUFFER_SIZE * sizeof(int32_t)));
}

template<int width, int height>
GpuSolver<width, height>::~GpuSolver() {
    ERR(cudaFree(GpuIndexesBuffer));
    ERR(cudaFree(GpuSegmentsBuffer));
}

template<int width, int height>
void GpuSolver<width, height>::GpuUp(uint32_t segment, uint32_t* indexes, uint32_t* out_segments, size_t count) {
    Timer timer;
    int threadsPerBlock = 256;
    int blocksPerGrid = ((int)count + threadsPerBlock - 1) / threadsPerBlock;
    ERR(cudaMemcpy(GpuIndexesBuffer, indexes, count * sizeof(int32_t), cudaMemcpyHostToDevice));
    kernel_up<width, height> << <blocksPerGrid, threadsPerBlock >> > (segment, GpuIndexesBuffer, GpuSegmentsBuffer, count);
    ERR(cudaGetLastError());
    ERR(cudaMemcpy(indexes, GpuIndexesBuffer, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ERR(cudaMemcpy(out_segments, GpuSegmentsBuffer, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    StatProcessedStates += count;
    StatExecutionNanos += timer.Elapsed();
}

template<int width, int height>
void GpuSolver<width, height>::GpuDown(uint32_t segment, uint32_t* indexes, uint32_t* out_segments, size_t count) {
    Timer timer;
    int threadsPerBlock = 256;
    int blocksPerGrid = ((int)count + threadsPerBlock - 1) / threadsPerBlock;
    ERR(cudaMemcpy(GpuIndexesBuffer, indexes, count * sizeof(int32_t), cudaMemcpyHostToDevice));
    kernel_dn<width, height> << <blocksPerGrid, threadsPerBlock >> > (segment, GpuIndexesBuffer, GpuSegmentsBuffer, count);
    ERR(cudaGetLastError());
    ERR(cudaMemcpy(indexes, GpuIndexesBuffer, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ERR(cudaMemcpy(out_segments, GpuSegmentsBuffer, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    StatProcessedStates += count;
    StatExecutionNanos += timer.Elapsed();
}

template<int width, int height>
void GpuSolver<width, height>::PrintStats() {
    std::cerr 
        << "Gpu: states=" << WithDecSep(StatProcessedStates)
        << "; time=" << WithTime(StatExecutionNanos) << std::endl;
}

template class GpuSolver<2, 2>;
template class GpuSolver<3, 2>;
template class GpuSolver<4, 2>;
template class GpuSolver<5, 2>;
template class GpuSolver<6, 2>;
template class GpuSolver<7, 2>;
template class GpuSolver<8, 2>;

template class GpuSolver<3, 3>;
template class GpuSolver<4, 3>;
template class GpuSolver<5, 3>;

template class GpuSolver<4, 4>;
