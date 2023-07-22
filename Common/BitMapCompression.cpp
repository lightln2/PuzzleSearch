#include "BitArray.h"
#include "BitMapCompression.h"
#include "Util.h"

#include <immintrin.h>

#include <vector>

size_t BitMapCompression::BitMapSize(int count, const uint32_t* indexes) {
    uint32_t first = indexes[0];
    uint32_t last = indexes[count - 1];
    if (last < first) return std::numeric_limits<size_t>::max();
    size_t bitmapSize = (last - first + 63) / 64;
    return bitmapSize * 8 + 8;
}

size_t BitMapCompression::EncodeBitMap(size_t count, const uint32_t* indexes, uint8_t* buffer) {
    uint32_t first = indexes[0];
    uint32_t last = indexes[count - 1];
    if (last < first) return 0;
    uint32_t bitmapSize = (last - first + 63) / 64;
    uint32_t bitmapSizeInBytes = bitmapSize * 8;
    *(uint32_t*)buffer = bitmapSize;
    *(uint32_t*)(buffer + 4) = first;
    uint64_t* bitmap = (uint64_t*)(buffer + 8);
    memset(bitmap, 0, bitmapSizeInBytes);
    for (int i = 1; i < count; i++) {
        uint32_t val = indexes[i] - first - 1;
        bitmap[val / 64] |= (1ui64 << (val % 64));
    }
    return 8ui64 + bitmapSizeInBytes;
}

size_t BitMapCompression::EncodeBitMapWithCheck(size_t count, const uint32_t* indexes, uint8_t* buffer) {
    uint32_t first = indexes[0];
    uint32_t last = indexes[count - 1];
    if (last < first) return 0;
    uint32_t max = last - first;
    uint32_t bitmapSize = (last - first + 63) / 64;
    uint32_t bitmapSizeInBytes = bitmapSize * 8;
    *(uint32_t*)buffer = bitmapSize;
    *(uint32_t*)(buffer + 4) = first;
    uint64_t* bitmap = (uint64_t*)(buffer + 8);
    memset(bitmap, 0, bitmapSizeInBytes);
    for (int i = 1; i < count; i++) {
        uint32_t val = indexes[i] - first;
        if (val > max) return -1;
        bitmap[val / 64] |= (1ui64 << ((val - 1) % 64));
    }
    return 8ui64 + bitmapSizeInBytes;
}

size_t BitMapCompression::DecodeBitMap(size_t& count, const uint8_t* buffer, uint32_t* indexes) {
    uint32_t bmsize = *(uint32_t*)buffer;
    uint32_t first = *(uint32_t*)(buffer + 4);
    uint64_t* bitmap = (uint64_t*)(buffer + 8);
    int dstPos = 0;
    indexes[dstPos++] = first;
    for (size_t i = 0; i < bmsize; i++) {
        ScanBits(bitmap[i], i * 64, [&](uint64_t index) {
            indexes[dstPos++] = first + int(index) + 1;
            });
    }
    //ensure(dstPos <= values_capacity);
    //size = 8 + bmsize * 8;
    count = dstPos;
    return 8 + bmsize * 8;
}
