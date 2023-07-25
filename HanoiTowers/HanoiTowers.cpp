#include "HanoiTowers.h"
#include "../Common/Util.h"

#include <immintrin.h>

template<int size>
struct FPState {
    uint64_t pegs[4] = { 0, 0, 0, 0 };
    int top[4] = { 255, 255, 255, 255 };

    void from_index(uint64_t index) {
        static constexpr uint64_t SIZE_MASK = (1ui64 << (2 * size)) - 1;
        static constexpr uint64_t BIT_MASK = 0x5555555555555555ui64 & SIZE_MASK;
        static const __m256i mask = _mm256_set1_epi64x(BIT_MASK);
        static const __m256i xmask = _mm256_set_epi64x(
            0,
            0x5555555555555555ui64,
            0xAAAAAAAAAAAAAAAAui64,
            -1);

        __m256i value = _mm256_set1_epi64x(index);
        __m256i maskedVal = _mm256_xor_si256(value, xmask);
        __m256i bitMasks =
            _mm256_and_si256(
                mask,
                _mm256_and_si256(
                    maskedVal,
                    _mm256_srli_epi64(maskedVal, 1)
                )
            );
        _mm256_storeu_epi64(pegs, bitMasks);

        /*
        uint64_t p0 = index >> 1;
        uint64_t p1 = index;
        uint64_t p0i = ~p0;
        uint64_t p1i = ~p1;
        pegs[0] = (p0i & p1i) & bitMask;
        pegs[1] = (p0i & p1) & bitMask;
        pegs[2] = (p0 & p1i) & bitMask;
        pegs[3] = (p0 & p1) & bitMask;
        */

        finish();
    }

    void add(int peg, int disk) {
        pegs[peg] |= (1ui64 << (2 * disk));
    }

    void finish() {
        unsigned long index;
        if (!_BitScanForward64(&index, pegs[0])) index = 255;
        top[0] = index;
        if (!_BitScanForward64(&index, pegs[1])) index = 255;
        top[1] = index;
        if (!_BitScanForward64(&index, pegs[2])) index = 255;
        top[2] = index;
        if (!_BitScanForward64(&index, pegs[3])) index = 255;
        top[3] = index;
    }

    uint64_t to_index() const {
        return pegs[1] | (pegs[2] << 1) | ((pegs[3] << 1) | pegs[3]);
    }

    bool empty(int peg) const {
        return pegs[peg] == 0;
    }

    bool has_disk(int peg, int disk) const {
        return pegs[peg] & (1ui64 << (2 * disk));
    }

    void move(int srcPeg, int dstPeg) {
        int disk = std::min(top[srcPeg], top[dstPeg]);
        if (disk != 255) {
            uint64_t bit = 1ui64 << disk;
            pegs[srcPeg] ^= bit;
            pegs[dstPeg] ^= bit;
        }
    }

    void restore_symmetry() {
        int bottom[4];
        for (int peg = 0; peg < 4; peg++) {
            unsigned long index;
            if (!_BitScanReverse64(&index, pegs[peg])) index = -1;
            bottom[peg] = index;
        }

        auto fn_restore = [&](int i, int j) {
            if (bottom[i] < bottom[j]) {
                std::swap(top[i], top[j]);
                std::swap(bottom[i], bottom[j]);
                std::swap(pegs[i], pegs[j]);
            }
        };
        fn_restore(2, 3);
        fn_restore(1, 2);
        fn_restore(2, 3);
    }

    static std::string StateToString(const FPState<size>& state) {
        std::ostringstream stream;
        for (int peg = 0; peg < 4; peg++) {
            for (int disk = 0; disk < size; disk++) {
                if (state.has_disk(peg, disk)) {
                    stream << (disk + 1) << ' ';
                }
            }
            stream << 0 << ' ';
        }
        auto str = stream.str();
        return str.substr(0, str.size() - 3);
    }

    static FPState<size> ParseState(const std::string& stateStr) {
        FPState<size> state;
        std::istringstream stream(stateStr);
        int disk = 0;
        int peg = 0;
        while (!stream.eof()) {
            stream >> disk;
            disk--;
            if (disk == -1) peg++;
            else state.add(peg, disk);
        }
        return state;
    }

};

template<int size>
std::string HanoiTowers<size>::ToString(uint64_t index) {
    FPState<size> state;
    state.from_index(index);
    return FPState<size>::StateToString(state);
}

template<int size>
uint64_t HanoiTowers<size>::Parse(std::string stateStr) {
    FPState<size> state = FPState<size>::ParseState(stateStr);
    return state.to_index();
}

template<int size>
void HanoiTowers<size>::Expand(uint64_t index, std::vector<uint64_t>& children) {
    FPState<size> state;
    state.from_index(index);
    auto fnMove = [&](int peg1, int peg2) {
        FPState<size> s2 = state;
        bool srcEmpty = s2.empty(peg1) || s2.empty(peg2);
        s2.move(peg1, peg2);
        bool dstEmpty = s2.empty(peg1) || s2.empty(peg2);
        if (srcEmpty || dstEmpty) {
            s2.restore_symmetry();
        }
        children.push_back(s2.to_index());
    };
    fnMove(0, 1);
    fnMove(0, 2);
    fnMove(0, 3);
    fnMove(1, 2);
    fnMove(1, 3);
    fnMove(2, 3);
}

template<int size>
void HanoiTowers<size>::ExpandCrossSegment(int segment, uint32_t index, std::vector<uint64_t>& children) {
    FPState<size> state;
    uint64_t segmentBase = uint64_t(segment) << 32;
    state.from_index(segmentBase | index);
    auto fnMove = [&](int peg1, int peg2) {
        FPState<size> s2 = state;
        bool srcEmpty = s2.empty(peg1) || s2.empty(peg2);
        s2.move(peg1, peg2);
        bool dstEmpty = s2.empty(peg1) || s2.empty(peg2);
        if (srcEmpty || dstEmpty) {
            s2.restore_symmetry();
        }
        uint64_t child = s2.to_index();
        if ((child >> 32) != segment) {
            children.push_back(child);
        }
    };
    fnMove(0, 1);
    fnMove(0, 2);
    fnMove(0, 3);
    fnMove(1, 2);
    fnMove(1, 3);
    fnMove(2, 3);
}

template<int size>
void HanoiTowers<size>::ExpandInSegment(int segment, uint32_t index, std::vector<uint32_t>& children) {
    FPState<size> state;
    state.from_index(uint64_t(segment) << 32 | index);
    bool noMovesBreakSymmetry =
        __popcnt64(state.pegs[1]) >= 2 &&
        __popcnt64(state.pegs[2]) >= 2 &&
        __popcnt64(state.pegs[3]) >= 1;
    //constexpr bool noMovesBreakSymmetry = false;

    if (noMovesBreakSymmetry) {
        auto fnMove = [&](int peg1, int peg2) {
            FPState<size> s2 = state;
            s2.move(peg1, peg2);
            uint64_t child = s2.to_index();
            if ((child >> 32) == segment) {
                children.push_back(uint32_t(child));
            }
        };
        fnMove(0, 1);
        fnMove(0, 2);
        fnMove(0, 3);
        fnMove(1, 2);
        fnMove(1, 3);
        fnMove(2, 3);
    }
    else {
        auto fnMove = [&](int peg1, int peg2) {
            FPState<size> s2 = state;
            bool srcEmpty = s2.empty(peg1) || s2.empty(peg2);
            s2.move(peg1, peg2);
            bool dstEmpty = s2.empty(peg1) || s2.empty(peg2);
            if (srcEmpty || dstEmpty) {
                s2.restore_symmetry();
            }
            uint64_t child = s2.to_index();
            if ((child >> 32) == segment) {
                children.push_back(uint32_t(child));
            }
        };
        fnMove(0, 1);
        fnMove(0, 2);
        fnMove(0, 3);
        fnMove(1, 2);
        fnMove(1, 3);
        fnMove(2, 3);
    }
}

template<int size>
void HanoiTowers<size>::ExpandInSegmentWithoutSmallest(int segment, uint32_t index, std::vector<uint32_t>& children) {
    int smallestDisk = index & 3;
    FPState<size> state;
    state.from_index(uint64_t(segment) << 32 | index);
    bool noMovesBreakSymmetry =
        __popcnt64(state.pegs[1]) >= 2 &&
        __popcnt64(state.pegs[2]) >= 2 &&
        __popcnt64(state.pegs[3]) >= 1;

    static int disksWithoutSmallest[4][3] = {
        {1, 2, 3},
        {0, 2, 3},
        {0, 1, 3},
        {0, 1, 2}
    };

    if (noMovesBreakSymmetry) {
        auto fnMove = [&](int peg1, int peg2) {
            FPState<size> s2 = state;
            s2.move(peg1, peg2);
            uint64_t child = s2.to_index();
            if ((child >> 32) == segment) {
                children.push_back(uint32_t(child));
            }
        };

        auto& dws = disksWithoutSmallest[smallestDisk];
        fnMove(dws[0], dws[1]);
        fnMove(dws[0], dws[2]);
        fnMove(dws[1], dws[2]);
    }
    else {
        auto fnMove = [&](int peg1, int peg2) {
            FPState<size> s2 = state;
            bool srcEmpty = s2.empty(peg1) || s2.empty(peg2);
            s2.move(peg1, peg2);
            bool dstEmpty = s2.empty(peg1) || s2.empty(peg2);
            if (srcEmpty || dstEmpty) {
                s2.restore_symmetry();
            }
            uint64_t child = s2.to_index();
            if ((child >> 32) == segment) {
                children.push_back(uint32_t(child));
            }
        };
        auto& dws = disksWithoutSmallest[smallestDisk];
        fnMove(dws[0], dws[1]);
        fnMove(dws[0], dws[2]);
        fnMove(dws[1], dws[2]);
    }
}

template<int size>
void HanoiTowers<size>::ExpandInSegmentNoSymmetry(int segment, uint32_t index, std::vector<uint32_t>& children) {
    FPState<size> state;
    state.from_index(uint64_t(segment) << 32 | index);
    auto fnMove = [&](int peg1, int peg2) {
        FPState<size> s2 = state;
        s2.move(peg1, peg2);
        uint64_t child = s2.to_index();
        if ((child >> 32) == segment) {
            children.push_back(uint32_t(child));
        }
    };
    fnMove(0, 1);
    fnMove(0, 2);
    fnMove(0, 3);
    fnMove(1, 2);
    fnMove(1, 3);
    fnMove(2, 3);
}

template<int size>
void HanoiTowers<size>::Expand(const std::vector<uint64_t>& indexes, std::vector<uint64_t>& children) {
    for (uint64_t index : indexes) {
        Expand(index, children);
    }
}

template<int size>
void HanoiTowers<size>::ExpandInSegment(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children) {
    for (size_t i = 0; i < count; i++) {
        ExpandInSegment(segment, indexes[i], children);
    }
}

template<int size>
void HanoiTowers<size>::ExpandInSegmentWithoutSmallest(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children) {
    for (size_t i = 0; i < count; i++) {
        ExpandInSegmentWithoutSmallest(segment, indexes[i], children);
    }
}

template<int size>
void HanoiTowers<size>::ExpandCrossSegment(int segment, const std::vector<uint32_t>& indexes, std::vector<uint64_t>& children) {
    uint64_t segmentBase = uint64_t(segment) << 32;

    for (uint64_t index : indexes) {
        ExpandCrossSegment(segment, index, children);
    }
}

template class HanoiTowers<14>;
template class HanoiTowers<15>;
template class HanoiTowers<16>;
template class HanoiTowers<17>;
template class HanoiTowers<18>;
template class HanoiTowers<19>;
template class HanoiTowers<20>;
template class HanoiTowers<21>;
template class HanoiTowers<22>;
template class HanoiTowers<23>;
template class HanoiTowers<24>;
