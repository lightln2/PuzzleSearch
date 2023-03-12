#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>

#define ensure(expression) do if(!(expression)) { \
            std::cerr << "ensure failed: " << __FILE__ << ":"<< __LINE__ << " " << #expression << std::endl; \
            assert(false); exit(-1); } while(0)

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

static std::string WithTime(uint64_t nanos) {
    std::ostringstream stream;
    auto millis = nanos / 1000000i64;
    auto seconds = millis / 1000;
    auto minutes = seconds / 60;
    auto hours = minutes / 60;
    stream 
        << std::setfill('0') 
        << std::setw(2) << hours << ':' 
        << std::setw(2) << minutes % 60 << ':' 
        << std::setw(2) << seconds % 60 << '.' 
        << std::setw(3) << millis % 1000;
    return stream.str();
}

static std::string WithSize(uint64_t size) {
    std::ostringstream stream;
    if (size < 1024) {
        stream << size << " bytes";
    }
    else if (size / 1024 < 1024) {
        stream << size / 1024 << " kb";
    }
    else if (size / 1024 / 1024 < 1024) {
        stream << size / 1024 / 1024 << " mb";
    }
    else {
        stream << size / 1024 / 1024 / 1024 << " gb";
    }
    return stream.str();
}


struct Timer {
    std::chrono::steady_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    uint64_t Elapsed() const { return (std::chrono::high_resolution_clock::now() - start).count(); }
};