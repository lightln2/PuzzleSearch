#pragma once

#include <cstdint>
#include <string>
#include <utility>

struct PuzzleState {
    int tiles[16];
    PuzzleState() = default;
    PuzzleState(int tiles[16]) { memcpy(this->tiles, tiles, sizeof(this->tiles)); }
    std::string ToString();
};

template<int width, int height>
class Puzzle {
public:
    static constexpr int size = width * height;
    static constexpr int BLANK_POS = 15;

public:
    static constexpr uint32_t TOTAL_INDEX_PER_SEGMENT() {
        uint32_t result = 16;
        for (int i = 3; i < std::min(13, size - 3); i++) {
            result *= i;
        }
        return result;
    }

    static bool CanMoveUp(uint32_t index) { return (index % 16) >= width; }
    static bool CanMoveDown(uint32_t index) { return (index % 16) < size - width; }
    static bool CanMoveLeft(uint32_t index) { return (index % 16) % width > 0; }
    static bool CanMoveRight(uint32_t index) { return (index % 16) % width < width - 1; }

    static uint32_t MoveLeft(uint32_t index) { return index - 1; }
    static uint32_t MoveRight(uint32_t index) { return index + 1; }

    static std::pair<uint32_t, uint32_t> MoveUp(std::pair<uint32_t, uint32_t> index) {
        return MoveUp(index.first, index.second);
    }

    static std::pair<uint32_t, uint32_t> MoveUp(uint32_t segment, uint32_t index) {
        return GetIndex(Pack(RotateUp(Unpack(FromIndex(segment, index)))));
    }

    static std::pair<uint32_t, uint32_t> MoveDown(std::pair<uint32_t, uint32_t> index) {
        return MoveDown(index.first, index.second);
    }

    static std::pair<uint32_t, uint32_t> MoveDown(uint32_t segment, uint32_t index) {
        return GetIndex(Pack(RotateDown(Unpack(FromIndex(segment, index)))));
    }

    static std::pair<uint32_t, uint32_t> Rank(std::string puzzle) {
        return GetIndex(Pack(Parse(puzzle)));
    }

    static std::string Unrank(std::pair<uint32_t, uint32_t> index) {
        return Unrank(index.first, index.second);
    }

    static std::string Unrank(uint32_t segment, uint32_t index) {
        return ToString(Unpack(FromIndex(segment, index)));
    }

    static PuzzleState Parse(std::string puzzle);
    static std::string ToString(PuzzleState puzzle);
    static std::string PrettyPrint(PuzzleState puzzle);

    static PuzzleState Pack(PuzzleState state);
    static PuzzleState Unpack(PuzzleState state);
    static bool Parity(PuzzleState state);

    static std::pair<uint32_t, uint32_t> GetIndex(PuzzleState state);
    static PuzzleState FromIndex(uint32_t segment, uint32_t index);

    static PuzzleState RotateUp(PuzzleState state);
    static PuzzleState RotateDown(PuzzleState state);
};
