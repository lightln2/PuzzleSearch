#pragma once

#include <iostream>
#include <cstdint>
#include <sstream>

#define ensure(expression) do if(!(expression)) { \
            std::cerr << "ensure failed: " << __FILE__ << ":"<< __LINE__ << " " << #expression << std::endl; \
            exit(-1); } while(0)

#define ENSURE_EQ(x, y) ensure((x)==(y))

static std::string WithDecSep(uint64_t value) {
    char str[32];
    int pos = 0;
    if (value == 0) return "0";
    while (value != 0) {
        if (pos % 4 == 3) str[pos++] = ',';
        str[pos++] = '0' + value % 10;
        value /= 10;
    }
    std::reverse(str, str + pos);
    return std::string(str, str + pos);
}
