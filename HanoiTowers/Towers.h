#pragma once

#include "../SlidingTilePuzzle/Util.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>


struct UnpackedHT {
    int heads[4];
    int8_t next[32];

    uint64_t Pack() {
        uint64_t index = 0;
        for (uint64_t h = 0; h < 4; h++) {
            for (int peg = heads[h]; peg >= 0; peg = next[peg]) {
                index |= (h << (peg * 2));
            }
        }
        return index;
    }

    void Unpack(int size, uint64_t index) {
        int tails[4]{ -1 };
        for (int h = 0; h < 4; h++) heads[h] = -1;
        for (int i = 0; i < size; i++) {
            int h = (index >> (i * 2)) & 3;
            if (tails[h] < 0) heads[h] = i;
            else next[tails[h]] = i;
            tails[h] = i;
            next[i] = -1;
        }
    }

    std::string ToString() {
        std::ostringstream stream;
        uint64_t index = 0;
        for (uint64_t h = 0; h < 4; h++) {
            if (h > 0) stream << ' ';
            stream << h << ": [ ";
            for (int peg = heads[h]; peg >= 0; peg = next[peg]) {
                stream << peg << ' ';
            }
            stream << "]";
        }
        return stream.str();
    }
};

// state: last 2 bits are dst peg of last move

template<int size>
class HanoiTowers {
public:
    static constexpr uint64_t InitialState = 1;
    static constexpr uint64_t InvalidState = 3;

    static std::string ToString(uint64_t state) {
        std::ostringstream stream;
        auto lastPeg = state % 4;
        auto index = state / 4;
        for (int i = 0; i < size; i++) {
            auto peg = (index >> (i * 2)) & 3;
            stream << peg << ' ';
        }
        stream << "last=" << lastPeg;
        return stream.str();
    }

private:
    UnpackedHT Unpack(uint64_t index) {
        UnpackedHT state;
        for (int i = 0; i < size; i++) {

        }
    }
};
