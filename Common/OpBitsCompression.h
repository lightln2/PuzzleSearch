#pragma once

#include <cstdint>

template<int BITS>
class OpBitsCompression {
public:
    static size_t Encode(size_t count, const uint8_t* opBits, uint8_t* buffer);
    static size_t Decode(size_t count, const uint8_t* buffer, uint8_t* opBits);
};

