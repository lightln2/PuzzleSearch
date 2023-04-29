#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

static void ERR(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

__device__ void GpuPermutationCompact(int* arr, int size) {
    int set_bits = 0;

    auto cntBits = [&](int index) {
        return __popcll(set_bits & ((1 << index) - 1));
    };

    for (int i = 0; i < size; i++) {
        int tile = arr[i];
        arr[i] -= cntBits(tile);
        set_bits |= (1 << tile);
    }
}

__device__ void GpuPermutationUncompact(int* arr, int size) {
    uint64_t tiles = 0xFEDCBA9876543210ui64;
    auto gettile = [&](int index) {
        return (int)(tiles >> (index * 4)) & 15;
    };
    auto removetile = [&](int index) {
        auto hi_tiles = (tiles >> (index * 4 + 4)) << (index * 4);
        auto lo_tiles = tiles & ((1ui64 << (index * 4)) - 1);
        tiles = hi_tiles | lo_tiles;
    };

    for (int i = 0; i < size; i++) {
        int tile = arr[i];
        arr[i] = gettile(tile);
        removetile(tile);
    }
}

__device__ uint64_t GpuPermutationRank(int* arr, int size) {
    GpuPermutationCompact(arr, size);

    uint64_t index = 0;
    for (int i = 0; i < size; i++) {
        index *= (size - i);
        index += arr[i];
    }

    return index;
}

__device__ void GpuPermutationUnrank(uint64_t index, int* arr, int size) {
    for (int i = size - 1; i >= 0; i--) {
        arr[i] = index % (size - i);
        index /= (size - i);
    }

    GpuPermutationUncompact(arr, size);
}


__global__ void kernel_test(uint64_t* indexes, int size, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int arr[16];
    GpuPermutationUnrank(indexes[i], arr, size);
    int temp = arr[size - 1];
    arr[size - 1] = arr[size - 2];
    arr[size - 2] = temp;
    indexes[i] = GpuPermutationRank(arr, size);
}

uint64_t* CreateGPUBuffer(int count) {
    uint64_t* gpuBuffer;
    ERR(cudaMalloc((void**)&gpuBuffer, count * sizeof(int64_t)));
    return gpuBuffer;
}

void DestroyGPUBuffer(uint64_t* gpuBuffer) {
    ERR(cudaFree(gpuBuffer));
}

void TestGpuPermutationRankUnrank(uint64_t* indexes, uint64_t* gpuBuffer, int size, int count) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    ERR(cudaMemcpy(gpuBuffer, indexes, count * sizeof(int64_t), cudaMemcpyHostToDevice));
    kernel_test<<<blocksPerGrid, threadsPerBlock>>>(gpuBuffer, size, count);
    ERR(cudaMemcpy(indexes, gpuBuffer, count * sizeof(int64_t), cudaMemcpyDeviceToHost));
    ERR(cudaGetLastError());
}
