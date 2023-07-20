#include "FrontierCompression.h"
#include "StreamVInt.h"
#include "BitArray.h"

namespace {
    constexpr uint32_t HI_BIT = 0x80000000;
}

namespace FrontierCompression {

    size_t BitMapSize(int count, uint32_t* indexes) {
        uint32_t first = indexes[0];
        uint32_t last = indexes[count - 1];
        if (last < first) return std::numeric_limits<size_t>::max();
        size_t bitmapSize = (last - first + 63) / 64;
        return bitmapSize * 8 + 8;
    }

    bool IsBitMap(uint8_t* buffer) {
        uint32_t bmsize = *(uint32_t*)buffer;
        return bmsize & HI_BIT;
    }

    int EncodeBitMap(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity) {
        uint32_t first = indexes[0];
        uint32_t last = indexes[count - 1];
        int bitmapSize = (last - first + 63) / 64;
        int bitmapSizeInBytes = bitmapSize * 8;
        *(uint32_t*)buffer = bitmapSize | HI_BIT;
        *(uint32_t*)(buffer + 4) = first;
        uint64_t* bitmap = (uint64_t*)(buffer + 8);
        memset(bitmap, 0, bitmapSizeInBytes);
        for (int i = 1; i < count; i++) {
            uint32_t val = indexes[i] - first - 1;
            bitmap[val / 64] |= (1ui64 << (val % 64));
        }
        return 8 + bitmapSizeInBytes;
    }

    int EncodeBitMapWithCheck(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity) {
        uint32_t first = indexes[0];
        uint32_t last = indexes[count - 1];
        if (last < first) return -1;
        uint32_t max = last - first - 1;
        int bitmapSize = (last - first + 63) / 64;
        int bitmapSizeInBytes = bitmapSize * 8;
        *(uint32_t*)buffer = bitmapSize | HI_BIT;
        *(uint32_t*)(buffer + 4) = first;
        uint64_t* bitmap = (uint64_t*)(buffer + 8);
        memset(bitmap, 0, bitmapSizeInBytes);
        for (int i = 1; i < count; i++) {
            uint32_t val = indexes[i] - first - 1;
            if (val > max) return -1;
            bitmap[val / 64] |= (1ui64 << (val % 64));
        }
        return 8 + bitmapSizeInBytes;
    }

    int DecodeBitMap(int& size, uint8_t* buffer, uint32_t* indexes, int values_capacity) {
        uint32_t bmsize = *(uint32_t*)buffer;
        ensure(bmsize & HI_BIT);
        bmsize &= ~HI_BIT;
        uint32_t first = *(uint32_t*)(buffer + 4);
        uint64_t* bitmap = (uint64_t*)(buffer + 8);
        int dstPos = 0;
        indexes[dstPos++] = first;
        for (int i = 0; i < bmsize; i++) {
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
        if (count > 128 && bitsPerHM < minBitsPerStreamVInt) {
            //std::cerr << "MAP: " << bitsPerHM << " < " << minBitsPerStreamVInt << std::endl;
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
            indexes.Size(),
            indexes.Buf(),
            buffer.Buf() + buffer.Size(),
            buffer.Capacity() - buffer.Size());
        buffer.SetSize(buffer.Size() + encoded);
    }

    void EncodeWithCheck(Buffer<uint32_t>& indexes, Buffer<uint8_t>& buffer) {
        size_t encoded = EncodeWithCheck(
            indexes.Size(),
            indexes.Buf(),
            buffer.Buf() + buffer.Size(),
            buffer.Capacity() - buffer.Size());
        buffer.SetSize(buffer.Size() + encoded);
    }

    int Decode(int position, Buffer<uint8_t>& buffer, Buffer<uint32_t>& indexes) {
        int size = buffer.Size() - position;
        size_t decoded = Decode(size, buffer.Buf() + position, indexes.Buf(), indexes.Capacity());
        indexes.SetSize(decoded);
        return position + size;
    }

}
