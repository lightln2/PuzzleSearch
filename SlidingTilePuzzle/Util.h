#pragma once

#include <iostream>
#include <cstdint>

#define ensure(expression) do if(!(expression)) { \
            std::cerr << __FILE__ << ":"<< __LINE__ << " " << #expression << std::endl; \
            exit(-1); } while(0)

#define ENSURE_EQ(x, y) ensure((x)==(y))
