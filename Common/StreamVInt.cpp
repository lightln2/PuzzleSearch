#include "StreamVInt.h"
#include "Util.h"

#include <immintrin.h>

std::atomic<uint64_t> StreamVInt::m_StatEncodeNanos(0);
std::atomic<uint64_t> StreamVInt::m_StatDecodeNanos(0);

struct SVEnc {
    __m128i shuffleMask;
    int length;
    uint8_t descriptor;
};

struct SVEncData {
    std::vector<SVEnc> result;
    __m128i preshuffle;
};

struct SVDec {
    __m128i shuffleMask;
    int length;
};

static SVEncData InitEncodingMask() {
    std::vector<SVEnc> result;
    result.reserve(65536);
    uint8_t data[16];
    for (int mask = 0; mask < (1 << 12); mask++) {
        SVEnc val;
        auto mbit = [&](int index) {
            return (mask >> index) & 1;
        };
        val.descriptor = 0;
        val.length = 1;
        int pos = 0;
        for (int i = 0; i < 4; i++) {
            if (mbit(3 * i + 2)) {
                val.descriptor |= 3 << (2 * i);
                val.length += 4;
                data[pos++] = 4 * i + 0;
                data[pos++] = 4 * i + 1;
                data[pos++] = 4 * i + 2;
                data[pos++] = 4 * i + 3;
            }
            else if (mbit(3 * i + 1)) {
                val.descriptor |= 2 << (2 * i);
                val.length += 3;
                data[pos++] = 4 * i + 0;
                data[pos++] = 4 * i + 1;
                data[pos++] = 4 * i + 2;
            }
            else if (mbit(3 * i + 0)) {
                val.descriptor |= 1 << (2 * i);
                val.length += 2;
                data[pos++] = 4 * i + 0;
                data[pos++] = 4 * i + 1;
            }
            else {
                val.descriptor |= 0 << (2 * i);
                val.length += 1;
                data[pos++] = 4 * i + 0;
            }
        }
        while (pos < 16) data[pos++] = 255;
        val.shuffleMask = _mm_loadu_epi8(data);
        result.push_back(val);
    }

    __m128i preshuffle = _mm_set_epi8(-1, -1, -1, -1, 15, 14, 13, 11, 10, 9, 7, 6, 5, 3, 2, 1);
    return { result, preshuffle };
}

static std::vector<SVDec> InitDecodingMask() {
    std::vector<SVDec> result;
    result.reserve(256);
    uint8_t data[16];
    for (int desc = 0; desc < 256; desc++) {
        SVDec val;
        val.length = 1;
        int pos = 0;
        for (int i = 0; i < 4; i++) {
            int bytes = 1 + (desc >> (i * 2)) % 4;
            val.length += bytes;
            if (bytes == 4) {
                data[4 * i + 0] = pos++;
                data[4 * i + 1] = pos++;
                data[4 * i + 2] = pos++;
                data[4 * i + 3] = pos++;
            }
            else if (bytes == 3) {
                data[4 * i + 0] = pos++;
                data[4 * i + 1] = pos++;
                data[4 * i + 2] = pos++;
                data[4 * i + 3] = -1;
            }
            else if (bytes == 2) {
                data[4 * i + 0] = pos++;
                data[4 * i + 1] = pos++;
                data[4 * i + 2] = -1;
                data[4 * i + 3] = -1;
            }
            else {
                data[4 * i + 0] = pos++;
                data[4 * i + 1] = -1;
                data[4 * i + 2] = -1;
                data[4 * i + 3] = -1;
            }
        }
        val.shuffleMask = _mm_loadu_epi8(data);
        result.push_back(val);
    }
    return result;
}

__forceinline static int _EncodeTuple(const uint32_t* indexes, uint8_t* buffer) {
    static SVEncData precalc = InitEncodingMask();
    __m128i data = _mm_loadu_epi32(indexes);
    __m128i pre_data = _mm_shuffle_epi8(data, precalc.preshuffle);
    __m128i cmpdata = _mm_cmpeq_epi8(pre_data, _mm_max_epu8(pre_data, _mm_set1_epi8(1)));
    int mask = _mm_movemask_epi8(cmpdata);
    auto& val = precalc.result[mask];
    __m128i result = _mm_shuffle_epi8(data, val.shuffleMask);
    buffer[0] = val.descriptor;
    *(__m128i*)(buffer + 1) = result;
    return val.length;
}

__forceinline static int _DecodeTuple(const uint8_t* buffer, uint32_t* indexes) {
    static std::vector<SVDec> precalc = InitDecodingMask();
    int descriptor = buffer[0];
    auto& val = precalc[descriptor];
    __m128i data = _mm_loadu_epi8(buffer + 1);
    __m128i result = _mm_shuffle_epi8(data, val.shuffleMask);
    _mm_storeu_epi8(indexes, result);
    return val.length;
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
        pos += _EncodeTuple(&indexes[i], &buffer[pos]);
    }
    return pos;
}

int StreamVInt::DecodeIndexes(int count, const uint8_t* buffer, uint32_t* indexes) {
    int pos = 0;
    for (int i = 0; i < count; i += 4) {
        pos += _DecodeTuple(&buffer[pos], &indexes[i]);
    }
    return pos;
}

int StreamVInt::DecodeIndexesAndDiff(int count, const uint8_t* buffer, uint32_t* indexes) {
    int pos = 0;
    uint32_t curIndex = 0;
    int i = 0;
    while (i < count) {
        pos += _DecodeTuple(&buffer[pos], &indexes[i]);
        curIndex = (indexes[i++] += curIndex);
        curIndex = (indexes[i++] += curIndex);
        curIndex = (indexes[i++] += curIndex);
        curIndex = (indexes[i++] += curIndex);
    }
    return pos;
}

int StreamVInt::EncodeBounds(int count, const uint8_t* bounds, uint8_t* buffer) {
    for (int i = 0; i < count; i += 8) {
        _EncodeBoundsTuple(&bounds[i], &buffer[i / 2]);
    }
    return count / 2;
}

int StreamVInt::DecodeBounds(int count, const uint8_t* buffer, uint8_t* bounds) {
    for (int i = 0; i < count; i += 8) {
        _DecodeBoundsTuple(&buffer[i / 2], &bounds[i]);
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

    srcPos += DecodeIndexesAndDiff(adjusted_count, &buffer[srcPos], &indexes[0]);

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
    ensure(dstPos <= buffer_capacity);
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

    srcPos += DecodeIndexesAndDiff(adjusted_count, &buffer[srcPos], &indexes[0]);
    srcPos += DecodeBounds(adjusted_count, &buffer[srcPos], &bounds[0]);

    size = srcPos;
    m_StatDecodeNanos += timer.Elapsed();
    return count;
}

void StreamVInt::PrintStats() {
    std::cerr
        << "StreamVInt: encode=" << WithTime(m_StatEncodeNanos)
        << "; decode=" << WithTime(m_StatDecodeNanos) << std::endl;
}
