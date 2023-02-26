#pragma once

#include <cstdint>
#include <string>

template<int width, int height>
class Puzzle {
public:
    static bool CanMoveUp(uint32_t index) {
        auto blank = index % 16;
        return blank >= width;
    }

    static bool CanMoveDown(uint32_t index) {
        auto blank = index % 16;
        return blank < width * height - width;
    }

    static bool CanMoveLeft(uint32_t index) {
        auto blank = index % 16;
        return (blank % width) > 0;
    }

    static bool CanMoveRight(uint32_t index) {
        auto blank = index % 16;
        return (blank % width) < width - 1;
    }



    static std::pair<uint32_t, uint32_t> GetIndex(std::string puzzle);
    static std::string FromIndex(uint32_t segment, uint32_t index);
};

