#pragma once

#include "Buffer.h"

namespace FrontierCompression {
    size_t BitMapSize(int count, uint32_t* indexes);
    int EncodeBitMap(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity);
    int EncodeBitMapWithCheck(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity);
    bool IsBitMap(uint8_t* buffer);
    int DecodeBitMap(int& size, uint8_t* buffer, uint32_t* indexes, int values_capacity);


    int Encode(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity);
    int EncodeWithCheck(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity);
    int Decode(int& size, uint8_t* buffer, uint32_t* indexes, int values_capacity);

    int Decode(int position, Buffer<uint8_t>& buffer, Buffer<uint32_t>& indexes);

    void Encode(Buffer<uint32_t>& indexes, Buffer<uint8_t>& buffer);
    void EncodeWithCheck(Buffer<uint32_t>& indexes, Buffer<uint8_t>& buffer);
}
