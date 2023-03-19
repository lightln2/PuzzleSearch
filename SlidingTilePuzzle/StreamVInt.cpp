#include "StreamVInt.h"
#include "Util.h"

#include <immintrin.h>

std::atomic<uint64_t> StreamVInt::m_StatEncodeNanos = 0;
std::atomic<uint64_t> StreamVInt::m_StatDecodeNanos = 0;

static struct SVEnc {
    __m128i shuffleMask;
    int length;
    uint8_t descriptor;
};

static std::vector<SVEnc> InitEncodingMask() {
    std::vector<SVEnc> result;
    result.reserve(65536);
    uint8_t data[16];
    for (int mask = 0; mask < 65536; mask++) {
        SVEnc val;
        auto mbit = [&](int index) {
            return (mask >> index) & 1;
        };
        val.descriptor = 0;
        val.length = 0;
        int pos = 0;
        if (mbit(3)) {
            val.descriptor |= 3;
            val.length += 4;
            data[pos++] = 0;
            data[pos++] = 1;
            data[pos++] = 2;
            data[pos++] = 3;
        }
        else if (mbit(2)) {
            val.descriptor |= 2;
            val.length += 3;
            data[pos++] = 0;
            data[pos++] = 1;
            data[pos++] = 2;
        }
        else if (mbit(1)) {
            val.descriptor |= 1;
            val.length += 2;
            data[pos++] = 0;
            data[pos++] = 1;
        }
        else {
            val.descriptor |= 0;
            val.length += 1;
            data[pos++] = 0;
        }

        if (mbit(7)) {
            val.descriptor |= 3 << 2;
            val.length += 4;
            data[pos++] = 4;
            data[pos++] = 5;
            data[pos++] = 6;
            data[pos++] = 7;
        }
        else if (mbit(6)) {
            val.descriptor |= 2 << 2;
            val.length += 3;
            data[pos++] = 4;
            data[pos++] = 5;
            data[pos++] = 6;
        }
        else if (mbit(5)) {
            val.descriptor |= 1 << 2;
            val.length += 2;
            data[pos++] = 4;
            data[pos++] = 5;
        }
        else {
            val.descriptor |= 0 << 2;
            val.length += 1;
            data[pos++] = 4;
        }

        if (mbit(11)) {
            val.descriptor |= 3 << 4;
            val.length += 4;
            data[pos++] = 8;
            data[pos++] = 9;
            data[pos++] = 10;
            data[pos++] = 11;
        }
        else if (mbit(10)) {
            val.descriptor |= 2 << 4;
            val.length += 3;
            data[pos++] = 8;
            data[pos++] = 9;
            data[pos++] = 10;
        }
        else if (mbit(9)) {
            val.descriptor |= 1 << 4;
            val.length += 2;
            data[pos++] = 8;
            data[pos++] = 9;
        }
        else {
            val.descriptor |= 0 << 4;
            val.length += 1;
            data[pos++] = 8;
        }

        if (mbit(15)) {
            val.descriptor |= 3 << 6;
            val.length += 4;
            data[pos++] = 12;
            data[pos++] = 13;
            data[pos++] = 14;
            data[pos++] = 15;
        }
        else if (mbit(14)) {
            val.descriptor |= 2 << 6;
            val.length += 3;
            data[pos++] = 12;
            data[pos++] = 13;
            data[pos++] = 14;
        }
        else if (mbit(13)) {
            val.descriptor |= 1 << 6;
            val.length += 2;
            data[pos++] = 12;
            data[pos++] = 13;
        }
        else {
            val.descriptor |= 0 << 6;
            val.length += 1;
            data[pos++] = 12;
        }

        while (pos < 16) data[pos++] = 255;
        val.shuffleMask = _mm_loadu_epi8(data);
        result.push_back(val);
    }
    return result;
}

__forceinline static int _EncodeTuple(const uint32_t* indexes, uint8_t* buffer) {
    static std::vector<SVEnc> precalc = InitEncodingMask();
    __m128i data = _mm_loadu_epi32(indexes);
    __m128i cmpdata = _mm_cmpeq_epi8(data, _mm_max_epu8(data, _mm_set1_epi8(1)));
    int mask = _mm_movemask_epi8(cmpdata);
    auto& val = precalc[mask];
    __m128i result = _mm_shuffle_epi8(data, val.shuffleMask);
    buffer[0] = val.descriptor;
    *(__m128i*)(buffer + 1) = result;
    return 1 + val.length;
}

__forceinline static int _DecodeTuple(const uint8_t* buffer, uint32_t* indexes) {
    int descriptor = buffer[0];
    int pos = 1;
    for (int i = 0; i < 4; i++) {
        int index = *(uint32_t*)&buffer[pos];

        int bytesCnt = 1 + ((descriptor >> (i * 2)) & 3);

        if (bytesCnt == 4) {
            pos += 4;
        }
        else if (bytesCnt == 3) {
            pos += 3;
            index &= 0xFFFFFF;
        }
        else if (bytesCnt == 2) {
            pos += 2;
            index &= 0xFFFF;
        }
        else {
            pos += 1;
            index &= 0xFF;
        }
        indexes[i] = index;
    }
    return pos;
}

__forceinline static void _EncodeBoundsTuple(const uint8_t* bounds, uint8_t* buffer) {
    uint64_t all = *(uint64_t*)bounds;
    *(uint32_t*)buffer = (uint32_t)(all | (all >> 28));
}

__forceinline static void _DecodeBoundsTuple(const uint8_t* buffer, uint8_t* bounds) {
    uint64_t stored = *(uint32_t*)buffer;
    *(uint64_t*)bounds = (stored & 0x0F0F0F0Fui64) | ((stored & 0xF0F0F0F0ui64) << 28);
}

int StreamVInt::EncodeTuple(const uint32_t* indexes, uint8_t* buffer) {
    return _EncodeTuple(indexes, buffer);
}

int StreamVInt::DecodeTuple(const uint8_t* buffer, uint32_t* indexes) {
    return _DecodeTuple(buffer, indexes);
}

void StreamVInt::EncodeBoundsTuple(const uint8_t* bounds, uint8_t* buffer) {
    _EncodeBoundsTuple(bounds, buffer);
}

void StreamVInt::DecodeBoundsTuple(const uint8_t* buffer, uint8_t* bounds) {
    return _DecodeBoundsTuple(buffer, bounds);
}

int StreamVInt::EncodeIndexes(int count, const uint32_t* indexes, uint8_t* buffer) {
    int pos = 0;
    for (int i = 0; i < count; i += 4) {
        pos += EncodeTuple(&indexes[i], &buffer[pos]);
    }
    return pos;
}

int StreamVInt::DecodeIndexes(int count, const uint8_t* buffer, uint32_t* indexes) {
    int pos = 0;
    for (int i = 0; i < count; i += 4) {
        pos += DecodeTuple(&buffer[pos], &indexes[i]);
    }
    return pos;
}

int StreamVInt::EncodeBounds(int count, const uint8_t* bounds, uint8_t* buffer) {
    for (int i = 0; i < count; i += 8) {
        EncodeBoundsTuple(&bounds[i], &buffer[i / 2]);
    }
    return count / 2;
}

int StreamVInt::DecodeBounds(int count, const uint8_t* buffer, uint8_t* bounds) {
    for (int i = 0; i < count; i += 8) {
        DecodeBoundsTuple(&buffer[i / 2], &bounds[i]);
    }
    return count / 2;
}

int StreamVInt::Encode(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity) {
    if (count == 0) return 0;
    Timer timer;
    int adjusted_count = (count + 7) / 8 * 8;
    assert(adjusted_count % 8 == 0);
    assert(adjusted_count <= MAX_INDEXES_COUNT);
    assert(buffer_capacity >= MAX_BUFFER_SIZE);

    for (int i = count - 1; i > 0; i--) {
        indexes[i] -= indexes[i - 1];
    }
    for (int i = count; i < adjusted_count; i++) {
        indexes[i] = 0;
    }

    int dstPos = 0;
    *(uint32_t*)&buffer[dstPos] = count;
    dstPos += 4;

    dstPos += EncodeIndexes(adjusted_count, &indexes[0], &buffer[dstPos]);
    m_StatEncodeNanos += timer.Elapsed();
    return dstPos;
}

int StreamVInt::Decode(int& size, uint8_t* buffer, uint32_t* indexes, int values_capacity) {
    if (size == 0) return 0;
    Timer timer;
    int srcPos = 0;
    int count = *(uint32_t*)&buffer[srcPos];
    srcPos += 4;

    int adjusted_count = (count + 7) / 8 * 8;
    assert(count > 0);
    assert(adjusted_count % 8 == 0);
    assert(adjusted_count <= MAX_INDEXES_COUNT);
    assert(values_capacity >= adjusted_count);

    srcPos += DecodeIndexes(adjusted_count, &buffer[srcPos], &indexes[0]);

    for (int i = 1; i < count; i++) {
        indexes[i] += indexes[i - 1];
    }

    size = srcPos;
    m_StatDecodeNanos += timer.Elapsed();
    return count;
}

int StreamVInt::Encode(int count, uint32_t* indexes, uint8_t* bounds, uint8_t* buffer, int buffer_capacity) {
    if (count == 0) return 0;
    Timer timer;
    int adjusted_count = (count + 7) / 8 * 8;
    assert(adjusted_count % 8 == 0);
    assert(adjusted_count <= MAX_INDEXES_COUNT);
    assert(buffer_capacity >= MAX_BUFFER_SIZE);

    for (int i = count - 1; i > 0; i--) {
        indexes[i] -= indexes[i - 1];
    }
    for (int i = count; i < adjusted_count; i++) {
        indexes[i] = 0;
        bounds[i] = 0;
    }

    int dstPos = 0;
    *(uint32_t*)&buffer[dstPos] = count;
    dstPos += 4;

    dstPos += EncodeIndexes(adjusted_count, &indexes[0], &buffer[dstPos]);
    dstPos += EncodeBounds(adjusted_count, &bounds[0], &buffer[dstPos]);
    m_StatEncodeNanos += timer.Elapsed();
    return dstPos;
}

int StreamVInt::Decode(int& size, uint8_t* buffer, uint32_t* indexes, uint8_t* bounds, int values_capacity) {
    if (size == 0) return 0;
    Timer timer;
    int srcPos = 0;
    int count = *(uint32_t*)&buffer[srcPos];
    srcPos += 4;

    int adjusted_count = (count + 7) / 8 * 8;
    assert(count > 0);
    assert(adjusted_count % 8 == 0);
    assert(adjusted_count <= MAX_INDEXES_COUNT);
    assert(values_capacity >= adjusted_count);

    srcPos += DecodeIndexes(adjusted_count, &buffer[srcPos], &indexes[0]);
    srcPos += DecodeBounds(adjusted_count, &buffer[srcPos], &bounds[0]);

    for (int i = 1; i < count; i++) {
        indexes[i] += indexes[i - 1];
    }

    size = srcPos;
    m_StatDecodeNanos += timer.Elapsed();
    return count;
}

void StreamVInt::PrintStats() {
    std::cerr 
        << "StreamVInt: encode=" << WithTime(m_StatEncodeNanos) 
        << "; decode=" << WithTime(m_StatDecodeNanos) << std::endl;
}
