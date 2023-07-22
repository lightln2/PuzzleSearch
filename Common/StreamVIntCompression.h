#pragma once

#include <cstdint>

class StreamVIntCompression {
public:
    static size_t Encode(size_t count, const uint32_t* indexes, uint8_t* buffer);
    static size_t Decode(size_t count, const uint8_t* buffer, uint32_t* indexes);
    static size_t DecodeWithDiff(size_t count, const uint8_t* buffer, uint32_t* indexes);
};

