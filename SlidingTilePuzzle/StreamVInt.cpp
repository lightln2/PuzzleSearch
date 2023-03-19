#include "StreamVInt.h"
#include "Util.h"

std::atomic<uint64_t> StreamVInt::m_StatEncodeNanos = 0;
std::atomic<uint64_t> StreamVInt::m_StatDecodeNanos = 0;

__forceinline static int _EncodeTuple(const uint32_t* indexes, uint8_t* buffer) {
    uint8_t descriptor = 0;
    int pos = 1;
    for (int i = 0; i < 4; i++) {
        uint32_t index = indexes[i];
        *(uint32_t*)&buffer[pos] = index;

        if (index >= 256 * 256 * 256) {
            descriptor |= (3 << (i * 2));
            pos += 4;
        }
        else if (index >= 256 * 256) {
            descriptor |= (2 << (i * 2));
            pos += 3;
        }
        else if (index >= 256) {
            descriptor |= (1 << (i * 2));
            pos += 2;
        }
        else {
            descriptor |= (0 << (i * 2));
            pos += 1;
        }
    }
    buffer[0] = descriptor;
    return pos;
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
    for (int i = 0; i < 4; i++) {
        buffer[i] = bounds[2 * i] | (bounds[2 * i + 1] * 16);
    }
}

__forceinline static void _DecodeBoundsTuple(const uint8_t* buffer, uint8_t* bounds) {
    for (int i = 0; i < 4; i++) {
        bounds[2 * i] = buffer[i] % 16;
        bounds[2 * i + 1] = buffer[i] / 16;
    }
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
