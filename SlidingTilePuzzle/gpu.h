#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

static void ERR(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
