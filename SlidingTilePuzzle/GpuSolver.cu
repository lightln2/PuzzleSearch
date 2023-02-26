#include "GpuSolver.h"
#include "gpu.h"

#include <sstream>

constexpr size_t OFFSET_ZERO = 15;

template <int size>
__device__ __forceinline__ void from_index(int* arr, uint64_t index) {
    arr[OFFSET_ZERO] = index & 15;
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
    char zeroPos = arr[OFFSET_ZERO];
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
            bool inv = (arr[j] >= arr[i]);
            arr[j] += (int)inv;
            invEven ^= inv;
        }
    }

    // restore by inversion count
    auto& zeroPos = arr[OFFSET_ZERO];
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
__global__ void kernel_up(uint32_t* indexes, uint32_t* segments, size_t count) {
    constexpr int size = width * height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int arr[16];
    from_index<size>(arr, ((uint64_t)segments[i] << 32) | indexes[i]);
    unpack<size, width, width % 2 == 0>(arr);
#ifdef _DEBUG
    if (!CanRotateUp<width>(arr)) {
        indexes[i] = (uint32_t)-1;
        segments[i] = (uint32_t)-1;
        return;
    }
#endif
    RotateUp<width>(arr);
    pack<size>(arr);
    uint64_t new_index = to_index<size>(arr);
    indexes[i] = new_index;
    segments[i] = new_index >> 32;
}

template<int width, int height>
__global__ void kernel_dn(uint32_t* indexes, uint32_t* segments, size_t count) {
    constexpr int size = width * height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int arr[16];
    from_index<size>(arr, ((uint64_t)segments[i] << 32) | indexes[i]);
    unpack<size, width, width % 2 == 0>(arr);
#ifdef _DEBUG
    if (!CanRotateDn<size, width>(arr)) {
        indexes[i] = (uint32_t)-1;
        segments[i] = (uint32_t)-1;
        return;
    }
#endif
    RotateDn<width>(arr);
    pack<size>(arr);
    uint64_t new_index = to_index<size>(arr);
    indexes[i] = new_index;
    segments[i] = new_index >> 32;
}

template<int width, int height>
__global__ void kernel_rank(uint32_t* input, uint32_t* output) {
    constexpr int size = width * height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0) return;
    int arr[16];
    for (int i = 0; i < 16; i++) arr[i] = input[i];
    pack<size>(arr);
    uint64_t new_index = to_index<size>(arr);
    output[0] = new_index >> 32;
    output[1] = new_index;
}

template<int width, int height>
__global__ void kernel_unrank(uint32_t segment, uint32_t index, uint32_t* output) {
    constexpr int size = width * height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0) return;
    int arr[16];
    from_index<size>(arr, ((uint64_t)segment << 32) | index);
    unpack<size, width, width % 2 == 0>(arr);
    for (int i = 0; i < 16; i++) output[i] = arr[i];
}


HostBuffer::HostBuffer() {
    ERR(cudaHostAlloc(&Buffer, GPU_BUFFER_SIZE * sizeof(int32_t), cudaHostAllocDefault));
}

HostBuffer::~HostBuffer() {
    ERR(cudaFreeHost(Buffer));
}

template<int width, int height>
uint64_t GpuSolver<width, height>::StatProcessedStates = 0;
template<int width, int height>
uint64_t GpuSolver<width, height>::StatExecutionMillis = 0;

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
void GpuSolver<width, height>::GpuUp(uint32_t* indexes, uint32_t* segments, size_t count) {
    int threadsPerBlock = 256;
    int blocksPerGrid = ((int)count + threadsPerBlock - 1) / threadsPerBlock;
    ERR(cudaMemcpy(GpuIndexesBuffer, indexes, count * sizeof(int32_t), cudaMemcpyHostToDevice));
    ERR(cudaMemcpy(GpuSegmentsBuffer, segments, count * sizeof(int32_t), cudaMemcpyHostToDevice));
    kernel_up<width, height> << <blocksPerGrid, threadsPerBlock >> > (GpuIndexesBuffer, GpuSegmentsBuffer, count);
    ERR(cudaGetLastError());
    ERR(cudaMemcpy(indexes, GpuIndexesBuffer, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ERR(cudaMemcpy(segments, GpuSegmentsBuffer, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
}

template<int width, int height>
void GpuSolver<width, height>::GpuDown(uint32_t* indexes, uint32_t* segments, size_t count) {
    int threadsPerBlock = 256;
    int blocksPerGrid = ((int)count + threadsPerBlock - 1) / threadsPerBlock;
    ERR(cudaMemcpy(GpuIndexesBuffer, indexes, count * sizeof(int32_t), cudaMemcpyHostToDevice));
    ERR(cudaMemcpy(GpuSegmentsBuffer, segments, count * sizeof(int32_t), cudaMemcpyHostToDevice));
    kernel_dn<width, height> << <blocksPerGrid, threadsPerBlock >> > (GpuIndexesBuffer, GpuSegmentsBuffer, count);
    ERR(cudaGetLastError());
    ERR(cudaMemcpy(indexes, GpuIndexesBuffer, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ERR(cudaMemcpy(segments, GpuSegmentsBuffer, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
}

template<int width, int height>
std::pair<uint32_t, uint32_t> GpuSolver<width, height>::Rank(std::string puzzle) {
    constexpr int size = width * height;
    int arr[16];
    std::istringstream stream(puzzle);
    int tile;
    int pos = 0;
    for (int i = 0; i < size; i++) {
        stream >> tile;
        if (tile == 0) arr[OFFSET_ZERO] = i;
        else arr[pos++] = tile - 1;
    }

    ERR(cudaMemcpy(GpuIndexesBuffer, arr, 16 * sizeof(int32_t), cudaMemcpyHostToDevice));
    kernel_rank<width, height> << <1, 1>> > (GpuIndexesBuffer, GpuSegmentsBuffer);
    ERR(cudaGetLastError());
    uint32_t outp[2];
    ERR(cudaMemcpy(outp, GpuSegmentsBuffer, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost));

    return { outp[0], outp[1] };
}

template<int width, int height>
std::string GpuSolver<width, height>::Unrank(uint32_t segment, uint32_t index) {
    constexpr int size = width * height;
    kernel_unrank<width, height> << <1, 1>> > (segment, index, GpuIndexesBuffer);
    ERR(cudaGetLastError());
    int arr[16];
    ERR(cudaMemcpy(arr, GpuIndexesBuffer, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost));

    std::ostringstream stream;
    int blank = arr[OFFSET_ZERO];
    for (int i = 0; i < size - 1; i++) {
        if (i > 0) stream << ' ';
        if (blank == i) stream << 0 << ' ';
        int tile = arr[i];
        stream << tile + 1;
    }
    if (blank == size - 1) stream << ' ' << 0;
    return stream.str();
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
