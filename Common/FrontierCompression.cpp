#include "FrontierCompression.h"
#include "StreamVInt.h"
#include "BitArray.h"

namespace {
    constexpr uint32_t CODEC_MAP = 0x80000000;
    constexpr uint32_t CODEC_ONEBYTE = 0x40000000;
    constexpr uint32_t CODEC_FOURBYTES = 0x20000000;
}

namespace FrontierCompression {

    bool IsBitMap(uint8_t* buffer) {
        uint32_t bmsize = *(uint32_t*)buffer;
        return bmsize & CODEC_MAP;
    }

    bool IsOneByte(uint8_t* buffer) {
        uint32_t bmsize = *(uint32_t*)buffer;
        return bmsize & CODEC_MAP;
    }

    bool IsFourBytes(uint8_t* buffer) {
        uint32_t bmsize = *(uint32_t*)buffer;
        return bmsize & CODEC_MAP;
    }

    size_t BitMapSize(int count, uint32_t* indexes) {
        uint64_t first = indexes[0];
        uint64_t last = indexes[count - 1];
        if (last < first) return std::numeric_limits<size_t>::max();
        size_t bitmapSize = (last - first + 63) / 64;
        return bitmapSize * 8 + 8;
    }

    int EncodeBitMap(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity) {
        uint64_t first = indexes[0];
        uint64_t last = indexes[count - 1];
        size_t bitmapSize = (last - first + 63) / 64;
        size_t bitmapSizeInBytes = bitmapSize * 8;
        ensure(bitmapSizeInBytes < 0xFFFFFFFF);
        *(uint32_t*)buffer = uint32_t(bitmapSize | CODEC_MAP);
        *(uint32_t*)(buffer + 4) = uint32_t(first);
        uint64_t* bitmap = (uint64_t*)(buffer + 8);
        memset(bitmap, 0, bitmapSizeInBytes);
        for (int i = 1; i < count; i++) {
            uint64_t val = indexes[i] - first - 1;
            bitmap[val / 64] |= (1ui64 << (val % 64));
        }
        return 8 + bitmapSizeInBytes;
    }

    int EncodeBitMapWithCheck(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity) {
        uint64_t first = indexes[0];
        uint64_t last = indexes[count - 1];
        // if last == first, most probably they are not all equal,
        // and cannot be map-encoded anyway
        if (last <= first) return -1;
        uint64_t max = last - first - 1;
        size_t bitmapSize = (last - first + 63) / 64;
        size_t bitmapSizeInBytes = bitmapSize * 8;
        *(uint32_t*)buffer = uint32_t(bitmapSize | CODEC_MAP);
        *(uint32_t*)(buffer + 4) = uint32_t(first);
        uint64_t* bitmap = (uint64_t*)(buffer + 8);
        memset(bitmap, 0, bitmapSizeInBytes);
        for (int i = 1; i < count; i++) {
            uint64_t val = indexes[i] - first - 1;
            if (val > max) return -1;
            bitmap[val / 64] |= (1ui64 << (val % 64));
        }
        return 8 + bitmapSizeInBytes;
    }

    int DecodeBitMap(int& size, uint8_t* buffer, uint32_t* indexes, int values_capacity) {
        uint32_t bmsize = *(uint32_t*)buffer;
        ensure(bmsize & CODEC_MAP);
        bmsize &= ~CODEC_MAP;
        uint32_t first = *(uint32_t*)(buffer + 4);
        uint64_t* bitmap = (uint64_t*)(buffer + 8);
        int dstPos = 0;
        indexes[dstPos++] = first;
        for (size_t i = 0; i < bmsize; i++) {
            ScanBits(bitmap[i], i * 64, [&](uint64_t index) {
                indexes[dstPos++] = first + int(index) + 1;
            });
        }
        ensure(dstPos <= values_capacity);
        size = 8 + bmsize * 8;
        return dstPos;
    }

    int Encode(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity) {
        if (count == 0) return 0;
        size_t minBitsPerStreamVInt = size_t(count) * 10;
        size_t bitsPerHM = BitMapSize(count, indexes) * 8;
        if (count > 128 && bitsPerHM < minBitsPerStreamVInt - 8) {
            return EncodeBitMap(count, indexes, buffer, buffer_capacity);
        }
        else {
            return StreamVInt::Encode(count, indexes, buffer, buffer_capacity);
        }
    }

    int EncodeWithCheck(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity) {
        if (count == 0) return 0;
        size_t minBitsPerStreamVInt = size_t(count) * 10;
        size_t bitsPerHM = BitMapSize(count, indexes) * 8;
        if (count > 128 && bitsPerHM < minBitsPerStreamVInt) {
            int encoded = EncodeBitMapWithCheck(count, indexes, buffer, buffer_capacity);
            if (encoded >= 0) {
                return encoded;
            }
        }
        return StreamVInt::Encode(count, indexes, buffer, buffer_capacity);
    }

    int Decode(int& size, uint8_t* buffer, uint32_t* indexes, int values_capacity) {
        if (size == 0) return 0;
        if (IsBitMap(buffer)) {
            return DecodeBitMap(size, buffer, indexes, values_capacity);
        }
        else {
            return StreamVInt::Decode(size, buffer, indexes, values_capacity);
        }
    }

    void Encode(Buffer<uint32_t>& indexes, Buffer<uint8_t>& buffer) {
        size_t encoded = Encode(
            int(indexes.Size()),
            indexes.Buf(),
            buffer.Buf() + buffer.Size(),
            int(buffer.Capacity() - buffer.Size()));
        buffer.SetSize(buffer.Size() + encoded);
    }

    void EncodeWithCheck(Buffer<uint32_t>& indexes, Buffer<uint8_t>& buffer) {
        size_t encoded = EncodeWithCheck(
            int(indexes.Size()),
            indexes.Buf(),
            buffer.Buf() + buffer.Size(),
            int(buffer.Capacity() - buffer.Size()));
        buffer.SetSize(buffer.Size() + encoded);
    }

    int Decode(int position, Buffer<uint8_t>& buffer, Buffer<uint32_t>& indexes) {
        int size = int(buffer.Size() - position);
        size_t decoded = Decode(size, buffer.Buf() + position, indexes.Buf(), int(indexes.Capacity()));
        indexes.SetSize(decoded);
        return position + size;
    }

}
