#pragma once

#include "Buffer.h"

#include <cstdint>

template<int BITS>
size_t EncodeOpBits(const uint8_t* src, size_t count, uint8_t* dst);

template<int BITS>
size_t DecodeOpBits(const uint8_t* src, size_t count, uint8_t* dst);

