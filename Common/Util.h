#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define ensure(expression) do if(!(expression)) { \
            std::cerr << "ensure failed: " << __FILE__ << ":"<< __LINE__ << " " << #expression << std::endl; \
            assert(false); exit(-1); } while(0)

#define ENSURE_EQ(x, y) ensure((x)==(y))

// converts number to tring with thousands separator : 1, 234, 567
std::string WithDecSep(uint64_t value);

std::string WithTime(uint64_t nanos);

std::string WithSize(uint64_t size);

void PrintVecor(const std::string& title, const std::vector<std::string>&values);

std::vector<std::string> CreatePaths(
    const std::vector<std::string>& directories,
    const std::string& path);

struct Timer {
    std::chrono::steady_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    uint64_t Elapsed() const { return (std::chrono::high_resolution_clock::now() - start).count(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
};

std::ostream& operator<<(std::ostream& os, const Timer& timer);
