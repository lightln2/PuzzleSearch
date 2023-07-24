 #include "PermutationMap.h"

#include <immintrin.h>
#include <vector>


void PermutationCompact(int* arr, int size) {
    int set_bits = 0;

    auto cntBits = [&](int index) {
        return __popcnt(set_bits & ((1<<index) - 1));
    };

    for (int i = 0; i < size; i++) {
        int tile = arr[i];
        arr[i] -= cntBits(tile);
        set_bits |= (1<<tile);
    }
}

void PermutationUncompact(int* arr, int size) {
    uint64_t tiles = 0xFEDCBA9876543210ui64;
    auto gettile = [&](int index) {
        return (int)(tiles >> (index * 4)) & 15;
    };
    auto removetile = [&](int index) {
        auto hi_tiles = (tiles >> (index * 4 + 4)) << (index * 4);
        auto lo_tiles = tiles & ((1ui64 << (index * 4)) - 1);
        tiles = hi_tiles | lo_tiles;
    };

    for (int i = 0; i < size; i++) {
        int tile = arr[i];
        arr[i] = gettile(tile);
        removetile(tile);
    }
}

uint64_t PermutationRank(int* arr, int size) {
    PermutationCompact(arr, size);

    uint64_t index = 0;
    for (int i = 0; i < size; i++) {
        index *= (size - i);
        index += arr[i];
    }

    return index;
}

void PermutationUnrank(uint64_t index, int* arr, int size) {
    for (int i = size - 1; i >= 0; i--) {
        arr[i] = index % (size - i);
        index /= (size - i);
    }

    PermutationUncompact(arr, size);
}
