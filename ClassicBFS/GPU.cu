#include "GPU.h"

#include <iostream>

void ERR(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

uint64_t* CreateGPUBuffer(int count) {
    uint64_t* gpuBuffer;
    ERR(cudaMalloc((void**)&gpuBuffer, count * sizeof(int64_t)));
    return gpuBuffer;
}

void DestroyGPUBuffer(uint64_t* gpuBuffer) {
    ERR(cudaFree(gpuBuffer));
}

CuStream CreateCudaStream() {
    cudaStream_t stream;
    ERR(cudaStreamCreate(&stream));
    return (CuStream)stream;
}

void DestroyCudaStream(CuStream stream) {
    ERR(cudaStreamDestroy(cudaStream_t(stream)));
}

void CopyToGpu(uint64_t* buffer, uint64_t* gpuBuffer, size_t count, CuStream stream) {
    ERR(cudaMemcpyAsync(gpuBuffer, buffer, count * sizeof(uint64_t), cudaMemcpyHostToDevice, cudaStream_t(stream)));
}

void CopyFromGpu(uint64_t* gpuBuffer, uint64_t* buffer, size_t count, CuStream stream) {
    ERR(cudaMemcpyAsync(buffer, gpuBuffer, count * sizeof(int64_t), cudaMemcpyDeviceToHost, cudaStream_t(stream)));
    ERR(cudaStreamSynchronize(cudaStream_t(stream)));
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
