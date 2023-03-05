#include "Puzzle.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <cassert>

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
static void ToStandardState(PuzzleState state, int arr[16]) {
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

std::string PuzzleState::ToString() {
    std::ostringstream stream;
    for (int i = 0; i < 15; i++) {
        stream << tiles[i] << ' ';
    }
    stream << "b=" << tiles[15];
    return stream.str();
}

template<int width, int height>
PuzzleState Puzzle<width, height>::Parse(std::string stateStr) {
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
    PuzzleState result{ tiles };
    assert(Parity(result));
    return result;
}

template<int width, int height>
std::string Puzzle<width, height>::ToString(PuzzleState state) {
    int arr[16]{ 255 };
    ToStandardState<size>(state, arr);

    std::ostringstream stream;
    for (int i = 0; i < size; i++) {
        if (i > 0) stream << ' ';
        stream << arr[i];
    }
    return stream.str();
}

template<int width, int height>
std::string Puzzle<width, height>::PrettyPrint(PuzzleState state) {
    int arr[16]{ 0 };
    ToStandardState<size>(state, arr);

    std::ostringstream stream;
    for (int i = 0; i < size; i++) {
        stream << std::setw(2) << arr[i] << ' ';
        if (i % width == width - 1) stream << '\n';
    }
    return stream.str();
}

template<int width, int height>
bool Puzzle<width, height>::Parity(PuzzleState state) {
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
PuzzleState Puzzle<width, height>::Pack(PuzzleState state) {
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
PuzzleState Puzzle<width, height>::Unpack(PuzzleState state) {
    int tiles[16]{ 0 };
    tiles[BLANK_POS] = state.tiles[BLANK_POS];
    for (int i = 0; i < size - 1; i++) {
        int x = state.tiles[i];
        for (int j = i + 1; j < size - 1; j++) {
            if (x >= state.tiles[j]) x++;
        }
        tiles[i] = x;
    }

    PuzzleState newstate{ tiles };
    
    if (!Parity(newstate)) {
        std::swap(newstate.tiles[0], newstate.tiles[1]);
    }
    assert(Parity(newstate));

    return newstate;
}

template<int width, int height>
std::pair<uint32_t, uint32_t> Puzzle<width, height>::GetIndex(PuzzleState state) {
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
PuzzleState Puzzle<width, height>::FromIndex(uint32_t segment, uint32_t index) {
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
PuzzleState Puzzle<width, height>::RotateUp(PuzzleState state) {
    PuzzleState newstate{ state.tiles };
    int blank = state.tiles[BLANK_POS];
    assert(CanMoveUp(blank));
    for (int i = blank - width; i < blank - 1; i++) {
        newstate.tiles[i] = state.tiles[i + 1];
    }
    newstate.tiles[blank - 1] = state.tiles[blank - width];
    newstate.tiles[BLANK_POS] = blank - width;
    return newstate;
}

template<int width, int height>
PuzzleState Puzzle<width, height>::RotateDown(PuzzleState state) {
    PuzzleState newstate{ state.tiles };
    int blank = state.tiles[BLANK_POS];
    assert(CanMoveDown(blank));
    for (int i = blank + width - 1; i > blank; i--) {
        newstate.tiles[i] = state.tiles[i - 1];
    }
    newstate.tiles[blank] = state.tiles[blank + width - 1];
    newstate.tiles[BLANK_POS] = blank + width;
    return newstate;
}

template class Puzzle<2, 2>;
template class Puzzle<3, 2>;
template class Puzzle<4, 2>;
template class Puzzle<5, 2>;
template class Puzzle<6, 2>;
template class Puzzle<7, 2>;
template class Puzzle<8, 2>;

template class Puzzle<3, 3>;
template class Puzzle<4, 3>;
template class Puzzle<5, 3>;

template class Puzzle<4, 4>;
