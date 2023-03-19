#include "StreamVInt.h"

int FrontierEncode(int& size, const uint32_t* indexes, const uint8_t* bounds, uint8_t* buffer, int capacity) {
    int srcPos = 0;
    int dstPos = 0;
    while (srcPos < size && dstPos + 5 <= capacity) {
        *(uint32_t*)&buffer[dstPos] = indexes[srcPos];
        dstPos += 4;
        buffer[dstPos++] = bounds[srcPos++];
    }
    size = srcPos;
    return dstPos;
}

int FrontierDecode(int& size, const uint8_t* buffer, uint32_t* indexes, uint8_t* bounds, int capacity) {
    int srcPos = 0;
    int dstPos = 0;
    while (srcPos < size && dstPos < capacity) {
        indexes[dstPos] = *(uint32_t*)&buffer[srcPos];
        srcPos += 4;
        bounds[dstPos++] = buffer[srcPos++];
    }
    size = srcPos;
    return dstPos;
}

int FrontierEncode(int pos, const Buffer<uint32_t>& indexes, const Buffer<uint8_t>& bounds, Buffer<uint8_t>& buffer) {
    int size = indexes.Size() - pos;
    int encoded = FrontierEncode(
        size,
        indexes.Buf() + pos,
        bounds.Buf() + pos,
        buffer.Buf() + buffer.Size(),
        buffer.Capacity() - buffer.Size());
    buffer.SetSize(buffer.Size() + encoded);
    return pos + size;
}

int FrontierDecode(int pos, const Buffer<uint8_t>& buffer, Buffer<uint32_t>& indexes, Buffer<uint8_t>& bounds) {
    int size = buffer.Size() - pos;
    int decoded = FrontierDecode(
        size,
        buffer.Buf() + pos,
        indexes.Buf() + indexes.Size(),
        bounds.Buf() + bounds.Size(),
        indexes.Capacity() - indexes.Size());
    indexes.SetSize(indexes.Size() + decoded);
    bounds.SetSize(bounds.Size() + decoded);
    return pos + size;
}

