#pragma once

#include <cstdint>

class BitMapCompression {
public:
    size_t BitMapSize(int count, const uint32_t* indexes);

    size_t EncodeBitMap(size_t count, const uint32_t* indexes, uint8_t* buffer);
    size_t EncodeBitMapWithCheck(size_t count, const uint32_t* indexes, uint8_t* buffer);
    size_t DecodeBitMap(size_t& count, const uint8_t* buffer, uint32_t* indexes);
};
