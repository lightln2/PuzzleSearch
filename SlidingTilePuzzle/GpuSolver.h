#pragma once

#include <atomic>
#include <cstdint>
#include <string>

constexpr size_t GPU_BUFFER_SIZE = 8 * 1024 * 1024;

struct HostBuffer {
    static constexpr size_t SIZE = GPU_BUFFER_SIZE;
    uint32_t* Buffer;

    HostBuffer();
    ~HostBuffer();
    HostBuffer(const HostBuffer&) = delete;
};

// TODO: use streams to run CUDA kernels in parallel

template<int width, int height>
class GpuSolver {
public:
    GpuSolver();
    ~GpuSolver();

    void GpuUp(uint32_t segment, uint32_t* indexes, uint32_t* out_segments, size_t count);
    void GpuDown(uint32_t segment, uint32_t* indexes, uint32_t* out_segments, size_t count);

    static void PrintStats();
private:

    uint32_t* GpuIndexesBuffer;
    uint32_t* GpuSegmentsBuffer;

    static std::atomic<uint64_t> StatProcessedStates;
    static std::atomic<uint64_t> StatExecutionNanos;
};
