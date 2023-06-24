#include "SlidingTilePuzzleOptimized.h"
#include "SlidingPuzzleGpu.h"
#include "PermutationMap.h"
#include "../Common/Util.h"
#include "GPU.h"

#include <sstream>

namespace {
    constexpr int BLANK_POS = 15;

    template<int size>
    constexpr int MaxSegments() {
        if (size == 16) return 0xEDC + 1;
        if (size == 15) return 0xDC + 1;
        if (size == 14) return 0xC + 1;
        return 1;
    }

    template<int size>
    constexpr uint32_t MaxIndexesPerSegment() {
        if (size >= 14) return 16U * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12;
        uint32_t max = 16;
        for (int i = 3; i < size; i++) max *= i;
        return max;
    }

    template<int size>
    static int ParseStandardState(std::string stateStr, int arr[16]) {
        std::istringstream stream(stateStr);
        int blank = -1;
        bool used[16]{ false };
        int tile;
        for (int i = 0; i < size; i++) {
            stream >> tile;
            assert(!used[tile]);
            used[tile] = true;
            assert(tile >= 0 && tile < size);
            if (tile == 0) blank = i;
            arr[i] = tile;
        }
        assert(blank >= 0 && blank < size);
        return blank;
    }

    template<int size>
    static void ToStandardState(const VPuzzleState& state, int arr[16]) {
        int blank = state.tiles[15];
        int pos = 0;
        for (int i = 0; i < size; i++) {
            if (i == blank) {
                arr[i] = 0;
                continue;
            }
            arr[i] = state.tiles[pos++] + 1;
        }
        assert(pos == size - 1);
    }

    template<int width, int height>
    VPuzzleState ParseState(std::string stateStr) {
        constexpr int size = width * height;
        int arr[16]{ 255 };
        int blank = ParseStandardState<size>(stateStr, arr);

        int tiles[16]{ 255 };
        int pos = 0;
        for (int i = 0; i < size; i++) {
            if (i == blank) continue;
            tiles[pos++] = arr[i] - 1;
        }
        assert(pos == size - 1);
        tiles[BLANK_POS] = blank;
        VPuzzleState result{ tiles };
        assert((Parity<width, height>(result)));
        return result;
    }

    template<int width, int height>
    bool Parity(VPuzzleState state) {
        constexpr int size = width * height;
        int inversions = 0;

        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < i; j++) {
                inversions += int(state.tiles[j] > state.tiles[i]);
            }
        }

        bool widthIsEven = width % 2 == 0;
        int row = state.tiles[BLANK_POS] / width;
        bool invIsEven = widthIsEven ?
            (row + inversions) % 2 == 0 :
            inversions % 2 == 0;

        return invIsEven;
    }

    template<int width, int height>
    VPuzzleState Pack(VPuzzleState state) {
        constexpr int size = width * height;
        int tiles[16]{ 0 };
        tiles[BLANK_POS] = state.tiles[BLANK_POS];
        for (int i = size - 2; i >= 0; i--) {
            int x = state.tiles[i];
            for (int j = i + 1; j < size - 1; j++) {
                if (state.tiles[i] > state.tiles[j]) x--;
            }
            tiles[i] = x;
        }
        for (int i = 0; i < size - 1; i++) {
            assert(tiles[i] >= 0 && tiles[i] <= i);
        }
        tiles[0] = 0;
        tiles[1] = 0;
        return { tiles };
    }

    template<int width, int height>
    VPuzzleState Unpack(VPuzzleState state) {
        constexpr int size = width * height;
        int tiles[16]{ 0 };
        tiles[BLANK_POS] = state.tiles[BLANK_POS];
        for (int i = 0; i < size - 1; i++) {
            int x = state.tiles[i];
            for (int j = i + 1; j < size - 1; j++) {
                if (x >= state.tiles[j]) x++;
            }
            tiles[i] = x;
        }

        VPuzzleState newstate{ tiles };

        if (!Parity<width, height>(newstate)) {
            std::swap(newstate.tiles[0], newstate.tiles[1]);
        }
        assert((Parity<width, height>(newstate)));

        return newstate;
    }

    template<int width, int height>
    std::pair<uint32_t, uint32_t> GetIndex(VPuzzleState state) {
        auto& tiles = state.tiles;
        uint32_t segment = (tiles[14] * 256) | (tiles[13] * 16) | tiles[12];
        uint32_t index =
            tiles[15] + 16 * (
                tiles[2] + 3 * (
                    tiles[3] + 4 * (
                        tiles[4] + 5 * (
                            tiles[5] + 6 * (
                                tiles[6] + 7 * (
                                    tiles[7] + 8 * (
                                        tiles[8] + 9 * (
                                            tiles[9] + 10 * (
                                                tiles[10] + 11 * tiles[11]
                                                )))))))));
        return { segment, index };
    }

    template<int width, int height>
    VPuzzleState FromIndex(uint32_t segment, uint32_t index) {
        int tiles[16]{ 0 };
        tiles[14] = segment / 256;
        tiles[13] = (segment / 16) % 16;
        tiles[12] = segment % 16;

        tiles[15] = index % 16;
        index /= 16;
        tiles[2] = index % 3;
        index /= 3;
        tiles[3] = index % 4;
        index /= 4;
        tiles[4] = index % 5;
        index /= 5;
        tiles[5] = index % 6;
        index /= 6;
        tiles[6] = index % 7;
        index /= 7;
        tiles[7] = index % 8;
        index /= 8;
        tiles[8] = index % 9;
        index /= 9;
        tiles[9] = index % 10;
        index /= 10;
        tiles[10] = index % 11;
        index /= 11;
        tiles[11] = index % 12;
        index /= 12;
        if (index > 0) throw std::runtime_error("index is not zero");
        return { tiles };
    }

    template<int width, int height>
    std::string ToString(VPuzzleState state) {
        constexpr int size = width * height;
        int arr[16]{ 255 };
        ToStandardState<size>(state, arr);

        std::ostringstream stream;
        for (int i = 0; i < size; i++) {
            if (i > 0) stream << ' ';
            stream << arr[i];
        }
        return stream.str();
    }

} // namespace

std::string VPuzzleState::ToString() {
    std::ostringstream stream;
    for (int i = 0; i < 15; i++) {
        stream << tiles[i] << ' ';
    }
    stream << "b=" << tiles[15];
    return stream.str();
}

template<int width, int height>
uint64_t SlidingTilePuzzleOptimized<width, height>::IndexesCount() const {
    uint64_t segments = MaxSegments<size>();
    return segments == 1 ?
        MaxIndexesPerSegment<size>() :
        segments << 32;
}

template<int width, int height>
std::string SlidingTilePuzzleOptimized<width, height>::ToString(uint64_t index) {
    VPuzzleState state = FromIndex<width, height>((index >> 32) & 0xFFFFFFFF, index & 0xFFFFFFFF);
    state = Unpack<width, height>(state);
    return state.ToString();
}

template<int width, int height>
uint64_t SlidingTilePuzzleOptimized<width, height>::Parse(std::string stateStr) {
    VPuzzleState state = ParseState<width, height>(stateStr);
    state = Pack<width, height>(state);
    auto [seg, idx] = GetIndex<width, height>(state);
    return (uint64_t(seg) << 32) | idx;
}

template<int width, int height>
void SlidingTilePuzzleOptimized<width, height>::ExpandGpu(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    uint64_t count,
    CuStream stream)
{
    GpuSlidingTilePuzzleOptimizedExpand<width, height>(gpuIndexes, gpuExpanded, count, stream);
}

template class SlidingTilePuzzleOptimized<2, 2>;
template class SlidingTilePuzzleOptimized<3, 2>;
template class SlidingTilePuzzleOptimized<4, 2>;
template class SlidingTilePuzzleOptimized<5, 2>;
template class SlidingTilePuzzleOptimized<6, 2>;
template class SlidingTilePuzzleOptimized<7, 2>;
template class SlidingTilePuzzleOptimized<8, 2>;

template class SlidingTilePuzzleOptimized<3, 3>;
template class SlidingTilePuzzleOptimized<4, 3>;
template class SlidingTilePuzzleOptimized<5, 3>;

template class SlidingTilePuzzleOptimized<4, 4>;
