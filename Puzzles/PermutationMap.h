#pragma once

#include <cstdint>

/*
improved version of Korf's linear time mapping between permutations and their indexes:
instead of pre-computed arrays it uses bit operations and CPU intrinsics
*/

void PermutationCompact(int* arr, int size);

void PermutationUncompact(int* arr, int size);


uint64_t PermutationRank(int* arr, int size);

void PermutationUnrank(uint64_t index, int* arr, int size);
