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

    static constexpr int B_UP = 1;
    static constexpr int B_DOWN = 2;
    static constexpr int B_LEFT = 4;
    static constexpr int B_RIGHT = 8;

public:
    /* segment number is always less than this */
    static constexpr int MaxSegments() {
        if (size == 16) return 0xEDC + 1;
        if (size == 15) return 0xDC + 1;
        if (size == 14) return 0xC + 1;
        return 1;
    }

    static constexpr uint32_t MaxIndexesPerSegment() {
        if (size >= 14) return 16U*3*4*5*6*7*8*9*10*11*12;
        uint32_t max = 16;
        for (int i = 3; i < size; i++) max *= i;
        return max;
    }

    static uint8_t GetBounds(int index) {
        uint8_t bound = 0;
        if (!CanMoveUp(index)) bound |= B_UP;
        if (!CanMoveDown(index)) bound |= B_DOWN;
        if (!CanMoveLeft(index)) bound |= B_LEFT;
        if (!CanMoveRight(index)) bound |= B_RIGHT;
        return bound;
    }

    static bool UpChangesSegment(int blank) {
        // MSVC does not do this optimization
        constexpr int mask = (1 << 13) | (1 << 14) | (1 << 15);
        return (mask >> blank) & 1;
        //return blank == 13 || blank == 14 || blank == 15;
    }

    static bool DownChangesSegment(int blank) {
        constexpr int mask = (1 << (13 - width)) | (1 << (14 - width)) | (1 << (15 - width));
        return (mask >> blank) & 1;
        //return blank == 13 - width || blank == 14 - width || blank == 15 - width;
    }

    static bool VerticalMoveChangesSegment(int blank) {
        constexpr int mask =
            (1 << 13) | (1 << 14) | (1 << 15) |
            (1 << (13 - width)) | (1 << (14 - width)) | (1 << (15 - width));
        return (mask >> blank) & 1;
    }

    static bool MultiTileHasCrossSegment(int blank) {
        if (size == 14) {
            return blank % width == 13 % width;
        }
        else if (size == 15) {
            return blank % width == 13 % width || blank % width == 14 % width;
        }
        else if (size == 16) {
            return blank % width == 13 % width || blank % width == 14 % width || blank % width == 15 % width;
        }
        else {
            return false;
        }
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
