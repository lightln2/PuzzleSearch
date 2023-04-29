#pragma once

#include <cstdint>

uint64_t* CreateGPUBuffer(int count);
void DestroyGPUBuffer(uint64_t* gpuBuffer);
void TestGpuPermutationRankUnrank(uint64_t* indexes, uint64_t* gpuBuffer, int size, int count);
