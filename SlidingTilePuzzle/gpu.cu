#include <ctime>

#include "gpu.h"

constexpr size_t MAX_COUNT = 4 * 1024 * 1024;
constexpr size_t OFFSET_ZERO = 15;


template <int size>
__device__ __forceinline__ void from_index(int* arr, int64_t index) {
    arr[15] = index & 15;
    index /= 16;

    int div = size;
    #pragma unroll
    for (int i = 0; i < size - 3; i++) {
        div--;
        int64_t newIndex = index / div;
        arr[i] = index - newIndex * div;
        index = newIndex;
    }
    arr[size - 3] = 0;
    arr[size - 2] = 0;
}

template <int size>
__device__ __forceinline__ int64_t to_index(int* arr) {
    char zeroPos = arr[15];
    int64_t index = 0;
    int div = 1;
    #pragma unroll
    for (int i = size - 3; i >= 0; i--)
    {
        div++;
        index = index * div + arr[i];
    }
    return index * 16 + zeroPos;
}

template <int size>
__device__ __forceinline__ void pack(int* arr) {
    #pragma unroll
    for (int i = 0; i < size - 4; i++) {
        #pragma unroll
        for (int j = i + 1; j < size - 3; j++) {
            arr[j] -= (int)(arr[j] >= arr[i]);
        }
    }
    arr[size - 2] = 0;
    arr[size - 3] = 0;
}

template <int size, int width, bool widthIsEven>
__device__ __forceinline__ void unpack(int* arr) {
    bool invEven = true;

    #pragma unroll
    for (int i = size - 2; i >= 0; i--) {
        #pragma unroll
        for (int j = i + 1; j < size - 1; j++) {
            bool q = (arr[j] >= arr[i]);
            arr[j] += (int)q;
            invEven ^= !q;
        }
    }

    // restore by inversion count
    auto& zeroPos = arr[15];
    bool rowEven = ((zeroPos / width) & 1) == 0;
    bool swapLast = (widthIsEven && invEven == rowEven) || (!widthIsEven && invEven);
    if (swapLast) {
        auto tmp = arr[size - 2];
        arr[size - 2] = arr[size - 3];
        arr[size - 3] = tmp;
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
__global__ void kernel_up(int32_t* indexes, int32_t* segments, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int arr[16];
    from_index<size>(arr, ((int64_t)segments[i] << 32) | data[i]);
    unpack<width * height, width, width % 2 == 0>(arr);
#ifdef _DEBUG
    if (!CanRotateUp<width>(arr)) {
        indexes[i] = -1;
        segments[i] = -1;
        return;
    }
#endif
    RotateUp<width>(arr);
    pack<size>(arr);
    data[i] = to_index<size>(arr);
}

template<int width, int height>
__global__ void kernel_dn(int32_t* indexes, int32_t* segments, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int arr[16];
    from_index<size>(arr, ((int64_t)segments[i] << 32) | data[i]);
    unpack<width* height, width, width % 2 == 0>(arr);
#ifdef _DEBUG
    if (!CanRotateDn<width>(arr)) {
        indexes[i] = -1;
        segments[i] = -1;
        return;
    }
#endif
    RotateDn<width>(arr);
    pack<size>(arr);
    data[i] = to_index<size>(arr);
}

template<int width, int height>
void gpu_up(uint32_t* indexes, uint32_t* gpu_indexes, uint32_t* segments, uint32_t* gpu_segments, size_t count) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_COUNT + threadsPerBlock - 1) / threadsPerBlock;

    ERR(cudaMemcpy(gpu_indexes, indexes, count * sizeof(int32_t), cudaMemcpyHostToDevice));
    ERR(cudaMemcpy(gpu_segments, segments, count * sizeof(int32_t), cudaMemcpyHostToDevice));
    kernel_up<width, height> << <blocksPerGrid, threadsPerBlock >> > (gpu_indexes, count);
    ERR(cudaGetLastError());
    ERR(cudaMemcpy(indexes, gpu_indexes, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ERR(cudaMemcpy(segments, gpu_segments, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
}

