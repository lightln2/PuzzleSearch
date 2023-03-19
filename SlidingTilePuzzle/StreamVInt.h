#pragma once

#include "FrontierFile.h"

#include <cstdint>

int FrontierEncode(int& size, const uint32_t* indexes, const uint8_t* bounds, uint8_t* buffer, int capacity);

int FrontierDecode(int& size, const uint8_t* buffer, uint32_t* indexes, uint8_t* bounds, int capacity);

int FrontierEncode(int pos, const Buffer<uint32_t>& indexes, const Buffer<uint8_t>& bounds, Buffer<uint8_t>& buffer);

int FrontierDecode(int pos, const Buffer<uint8_t>& buffer, Buffer<uint32_t>& indexes, Buffer<uint8_t>& bounds);
