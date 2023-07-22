#include "OpBitsCompression.h"
#include "Util.h"

template<int BITS>
__forceinline static int _EncodeTuple(const uint8_t* opBits, uint8_t* buffer);

template<int BITS>
__forceinline static int _DecodeTuple(const uint8_t* buffer, uint8_t* opBits);

template<>
__forceinline static int _EncodeTuple<2>(const uint8_t* opBits, uint8_t* buffer) {
    uint64_t all = *(uint64_t*)opBits;
    *(uint16_t*)buffer = (uint16_t)(all | (all >> 14) | (all >> 28) | (all >> 42));
    return 2;
}

template<>
__forceinline static int _DecodeTuple<2>(const uint8_t* buffer, uint8_t* opBits) {
    uint64_t stored = *(uint16_t*)buffer;
    *(uint64_t*)opBits =
        (stored & 0x0303ui64) |
        ((stored & 0x0C0Cui64) << 14) |
        ((stored & 0x3030ui64) << 28) |
        ((stored & 0xC0C0ui64) << 42);
    return 2;
}

template<>
__forceinline static int _EncodeTuple<4>(const uint8_t* opBits, uint8_t* buffer) {
    uint64_t all = *(uint64_t*)opBits;
    *(uint32_t*)buffer = (uint32_t)(all | (all >> 28));
    return 4;
}

template<>
__forceinline static int _DecodeTuple<4>(const uint8_t* buffer, uint8_t* opBits) {
    uint64_t stored = *(uint32_t*)buffer;
    *(uint64_t*)opBits = (stored & 0x0F0F0F0Fui64) | ((stored & 0xF0F0F0F0ui64) << 28);
    return 4;
}

template<>
__forceinline static int _EncodeTuple<8>(const uint8_t* opBits, uint8_t* buffer) {
    *(uint64_t*)buffer = *(uint64_t*)opBits;
    return 8;
}

template<>
__forceinline static int _DecodeTuple<8>(const uint8_t* buffer, uint8_t* opBits) {
    *(uint64_t*)opBits = *(uint64_t*)buffer;
    return 8;
}

template<int BITS>
size_t OpBitsCompression<BITS>::Encode(size_t count, const uint8_t* opBits, uint8_t* buffer) {
    ensure((count & 7) == 0);
    size_t pos = 0;
    for (size_t i = 0; i < count; i += 8) {
        pos += _EncodeTuple<BITS>(&opBits[i], &buffer[pos]);
    }
    return pos;
}

template<int BITS>
size_t OpBitsCompression<BITS>::Decode(size_t count, const uint8_t* buffer, uint8_t* opBits) {
    ensure((count & 7) == 0);
    size_t pos = 0;
    for (size_t i = 0; i < count; i += 8) {
        pos += _DecodeTuple<BITS>(&buffer[pos], &opBits[i]);
    }
    return pos;
}

template class OpBitsCompression<2>;
template class OpBitsCompression<4>;
template class OpBitsCompression<8>;
