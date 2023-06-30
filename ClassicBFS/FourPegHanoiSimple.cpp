#include "FourPegHanoiSimple.h"
#include "../Common/Util.h"

#include <cstdint>

namespace {

    bool HasOp(int op, int dstDisk) {
        return op & (1 << dstDisk);
    }

    struct FPState {
        int next[32] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
        int top[4] = { -1, -1, -1, -1 };
        int bottom[4] = { -1, -1, -1, -1 };

        void add(int peg, int disk) {
            if (top[peg] == -1) top[peg] = disk;
            else next[bottom[peg]] = disk;
            bottom[peg] = disk;
        }

        bool can_move(int srcPeg, int dstPeg) {
            if (srcPeg == dstPeg) return false;
            int srcDisk = top[srcPeg];
            if (srcDisk == -1) return false;
            int dstDisk = top[dstPeg];
            if (dstDisk != -1 && dstDisk < srcDisk) return false;
            return true;
        }

        bool move(int srcPeg, int dstPeg) {
            if (srcPeg == dstPeg) return false;
            int srcDisk = top[srcPeg];
            if (srcDisk == -1) return false;
            int dstDisk = top[dstPeg];
            if (dstDisk != -1 && dstDisk < srcDisk) return false;
            top[srcPeg] = next[srcDisk];
            if (top[srcPeg] == -1) bottom[srcPeg] = -1;
            if (dstDisk == -1) {
                top[dstPeg] = bottom[dstPeg] = srcDisk;
                next[srcDisk] = -1;
            }
            else {
                top[dstPeg] = srcDisk;
                next[srcDisk] = dstDisk;
            }
            return true;
        }

        int restore_symmetry(int dstPeg) {
            int pegs[4]{ 0, 1, 2, 3 };
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
            int invpegs[4];
            invpegs[pegs[0]] = 0;
            invpegs[pegs[1]] = 1;
            invpegs[pegs[2]] = 2;
            invpegs[pegs[3]] = 3;
            return invpegs[dstPeg];
        }
    };

    std::string StateToString(const FPState& state) {
        std::ostringstream stream;
        for (int peg = 0; peg < 4; peg++) {
            for (int disk = state.top[peg]; disk >= 0; disk = state.next[disk]) {
                stream << (disk + 1) << ' ';
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
        int lastDisk = -1;
        int peg = 0;
        while (!stream.eof()) {
            stream >> disk;
            disk--;
            if (disk == -1) {
                lastDisk = -1;
                peg++;
            }
            else {
                state.add(peg, disk);
            }
        }
        return state;
    }

    uint64_t StateToIndex(const FPState& state) {
        uint64_t index = 0;
        for (int peg = 0; peg < 4; peg++) {
            int top = state.top[peg];
            for (int disk = top; disk >= 0; disk = state.next[disk]) {
                index |= (uint64_t(peg) << (2 * disk));
            }
        }
        return index;
    }

    FPState IndexToState(int size, uint64_t index) {
        FPState state;
        for (int i = 0; i < 32; i++) state.next[i] = -1;
        for (int i = 0; i < 4; i++) state.top[i] = -1;
        for (int disk = 0; disk < size; disk++) {
            int peg = (index >> (2 * disk)) & 3;
            state.add(peg, disk);
        }
        return state;
    }

} // namespace

std::string FourPegHanoiSimple::Name() const { 
    std::ostringstream stream;
    stream
        << "Four-Peg Hanoi Towers, size=" << m_Size
        << "; symmetry=" << m_UseSymmetry;
    return stream.str();
}

FourPegHanoiSimple::FourPegHanoiSimple(int size, bool useSymmetry)
    : m_Size(size)
    , m_UseSymmetry(useSymmetry)
{}

uint64_t FourPegHanoiSimple::IndexesCount() const {
    return 1ui64 << (m_Size * 2);
}


std::string FourPegHanoiSimple::ToString(uint64_t index) {
    auto state = IndexToState(m_Size, index);
    return StateToString(state);
}

uint64_t FourPegHanoiSimple::Parse(std::string stateStr) {
    FPState state = ParseState(stateStr);
    return StateToIndex(state);
}

void FourPegHanoiSimple::Expand(uint64_t index, int op, uint64_t* children, int* operators) {
    FPState state = IndexToState(m_Size, index);
    int pos = 0;
    for (int peg1 = 0; peg1 < 4; peg1++) {
        for (int peg2 = peg1 + 1; peg2 < 4; peg2++) {
            FPState s2 = state;
            int p1 = peg1, p2 = peg2;
            if (!s2.can_move(p1, p2)) std::swap(p1, p2);
            if (s2.move(p1, p2)) {
                if (m_UseSymmetry) {
                    p2 = s2.restore_symmetry(p2);
                }
                if (!HasOp(op, p1)) {
                    children[pos] = StateToIndex(s2);
                    operators[pos] = p2;
                }
            }
            pos++;
        }
    }
}

void FourPegHanoiSimple::Expand(
    std::vector<uint64_t>& indexes,
    std::vector<int>& usedOperatorBits,
    std::vector<uint64_t>& expandedIndexes,
    std::vector<int>& expandedOperators,
    ExpandHint hint)
{
    if (expandedIndexes.capacity() < MAX_INDEXES_BUFFER * 6) {
        expandedIndexes.reserve(MAX_INDEXES_BUFFER * 6);
    }
    if (expandedOperators.capacity() < MAX_INDEXES_BUFFER * 6) {
        expandedOperators.reserve(MAX_INDEXES_BUFFER * 6);
    }
    expandedIndexes.clear();
    expandedOperators.clear();
    expandedIndexes.resize(indexes.size() * 6, INVALID_INDEX);
    expandedOperators.resize(indexes.size() * 6, -1);

    for (int i = 0; i < indexes.size(); i++) {
        Expand(indexes[i], usedOperatorBits[i], &expandedIndexes[6 * i], &expandedOperators[6 * i]);
    }
}