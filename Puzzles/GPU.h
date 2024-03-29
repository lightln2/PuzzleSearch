#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>

using CuStream = void*;

void ERR(cudaError_t err);

uint64_t* CreateGPUBuffer(int count);
uint32_t* CreateGPUBuffer32(int count);
void DestroyGPUBuffer(uint64_t* gpuBuffer);
void DestroyGPUBuffer32(uint32_t* gpuBuffer);

CuStream CreateCudaStream();
void DestroyCudaStream(CuStream stream);

void CopyToGpu(const uint64_t* buffer, uint64_t* gpuBuffer, size_t count, CuStream stream);
void CopyFromGpu(const uint64_t* gpuBuffer, uint64_t* buffer, size_t count, CuStream stream);

void CopyToGpu(const uint32_t* buffer, uint32_t* gpuBuffer, size_t count, CuStream stream);
void CopyFromGpu(const uint32_t* gpuBuffer, uint32_t* buffer, size_t count, CuStream stream);

__device__ void GpuPermutationCompact(int* arr, int size);

__device__ void GpuPermutationUncompact(int* arr, int size);

__device__ uint64_t GpuPermutationRank(int* arr, int size);

__device__ void GpuPermutationUnrank(uint64_t index, int* arr, int size);
