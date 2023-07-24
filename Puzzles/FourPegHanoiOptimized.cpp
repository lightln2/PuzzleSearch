#include "FourPegHanoiOptimized.h"
#include "../Common/Util.h"

#include <immintrin.h>

#include <cstdint>

namespace {

    bool HasOp(int op, int dstDisk) {
        return op & (1 << dstDisk);
    }

    struct FPState {
        uint64_t pegs[4] = { 0, 0, 0, 0 };
        int top[4] = { 255, 255, 255, 255 };

        void from_index(uint64_t index, int size) {
            const uint64_t SIZE_MASK = (1ui64 << (2 * size)) - 1;
            const uint64_t BIT_MASK = 0x5555555555555555ui64 & SIZE_MASK;
            from_index_bm(index, BIT_MASK);
        }

        void from_index_bm(uint64_t index, uint64_t bitMask) {
            
            __m256i value = _mm256_set1_epi64x(index);
            __m256i xmask = _mm256_set_epi64x(
                0,
                0x5555555555555555ui64,
                0xAAAAAAAAAAAAAAAAui64,
                -1);
            __m256i maskedVal = _mm256_xor_si256(value, xmask);
            __m256i mask = _mm256_set1_epi64x(bitMask);
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
            for (int peg = 0; peg < 4; peg++) {
                unsigned long index;
                if (!_BitScanForward64(&index, pegs[peg])) index = 255;
                top[peg] = index;
            }
        }

        uint64_t to_index() const {
            return pegs[1] | (pegs[2] * 2) | (pegs[3] * 3);
        }

        bool empty(int peg) const {
            return pegs[peg] == 0;
        }

        bool can_move(int srcPeg, int dstPeg) const {
            return top[srcPeg] < top[dstPeg];
        }

        bool has_disk(int peg, int disk) const {
            return pegs[peg] & (1ui64 << (2 * disk));
        }

        bool move(int srcPeg, int dstPeg) {
            if (!can_move(srcPeg, dstPeg)) return false;
            pegs[srcPeg] -= (1ui64 << top[srcPeg]);
            pegs[dstPeg] |= (1ui64 << top[srcPeg]);
            return true;
        }

        int restore_symmetry(int dstPeg) {
            int bottom[4];
            for (int peg = 0; peg < 4; peg++) {
                unsigned long index;
                if (!_BitScanReverse64(&index, pegs[peg])) index = -1;
                bottom[peg] = index;
            }

            int pegIndexes[4]{ 0, 1, 2, 3 };
            auto fn_restore = [&](int i, int j) {
                if (bottom[i] < bottom[j]) {
                    std::swap(top[i], top[j]);
                    std::swap(bottom[i], bottom[j]);
                    std::swap(pegs[i], pegs[j]);
                    std::swap(pegIndexes[i], pegIndexes[j]);
                }
            };
            fn_restore(2, 3);
            fn_restore(1, 2);
            fn_restore(2, 3);
            int invpegs[4];
            invpegs[pegIndexes[0]] = 0;
            invpegs[pegIndexes[1]] = 1;
            invpegs[pegIndexes[2]] = 2;
            invpegs[pegIndexes[3]] = 3;
            return invpegs[dstPeg];
        }
    };

    std::string StateToString(const FPState& state, int size) {
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

    FPState ParseState(const std::string& stateStr) {
        FPState state;
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

} // namespace

std::string FourPegHanoiOptimized::Name() const {
    std::ostringstream stream;
    stream
        << "Four-Peg Hanoi Towers Optimized, size=" << m_Size
        << "; symmetry=" << m_UseSymmetry;
    return stream.str();
}

FourPegHanoiOptimized::FourPegHanoiOptimized(int size, bool useSymmetry)
    : m_Size(size)
    , m_UseSymmetry(useSymmetry)
{}

uint64_t FourPegHanoiOptimized::IndexesCount() const {
    return 1ui64 << (m_Size * 2);
}


std::string FourPegHanoiOptimized::ToString(uint64_t index) {
    FPState state;
    state.from_index(index, m_Size);
    return StateToString(state, m_Size);
}

uint64_t FourPegHanoiOptimized::Parse(std::string stateStr) {
    FPState state = ParseState(stateStr);
    return state.to_index();
}

void FourPegHanoiOptimized::Expand(uint64_t index, int op, std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators) {
    FPState state;
    static const uint64_t SIZE_MASK = (1ui64 << (2 * m_Size)) - 1;
    static const uint64_t BIT_MASK = 0x5555555555555555ui64 & SIZE_MASK;

    state.from_index(index, m_Size);
    //state.from_index_bm(index, BIT_MASK);
    for (int peg1 = 0; peg1 < 4; peg1++) {
        for (int peg2 = peg1 + 1; peg2 < 4; peg2++) {
            FPState s2 = state;
            int p1 = peg1, p2 = peg2;
            if (!s2.can_move(p1, p2)) std::swap(p1, p2);
            bool dstWasEmpty = s2.empty(p2);
            if (s2.move(p1, p2)) {
                bool srcIsEmpty = s2.empty(p1);
                if (m_UseSymmetry && (dstWasEmpty || srcIsEmpty)) {
                    p2 = s2.restore_symmetry(p2);
                }
                if (!HasOp(op, p1)) {
                    expandedIndexes.push_back(s2.to_index());
                    expandedOperators.push_back(p2);
                }
            }
        }
    }
}

void FourPegHanoiOptimized::Expand(
    std::vector<uint64_t>& indexes,
    std::vector<int>& usedOperatorBits,
    std::vector<uint64_t>& expandedIndexes,
    std::vector<int>& expandedOperators,
    ExpandHint hint)
{
    SetupOutputBuffers(expandedIndexes, expandedOperators);

    bool filterXSeg = (hint.SegmentBits == 32 && hint.CrossSegment && m_Size > 16);

    auto fbInSegOnly = [&](uint64_t index) {
        uint32_t idx = index & 0xFFFFFFFF;
        uint32_t p0 = idx >> 1;
        uint32_t p1 = idx;
        uint32_t p0i = ~p0;
        uint32_t p1i = ~p1;
        bool z0 = ((p0 & p1) & 0x55555555ui32) != 0;
        bool z1 = ((p0 & p1i) & 0x55555555ui32) != 0;
        bool z2 = ((p0i & p1) & 0x55555555ui32) != 0;
        bool z3 = ((p0i & p1i) & 0x55555555ui32) != 0;
        return int(z0) + int(z1) + int(z2) + int(z3) >= 3;
    };

    for (int i = 0; i < indexes.size(); i++) {
        if (filterXSeg) {
            if (fbInSegOnly(indexes[i])) continue;
        }
        size_t start = expandedIndexes.size();
        Expand(indexes[i], usedOperatorBits[i], expandedIndexes, expandedOperators);
    }
}
