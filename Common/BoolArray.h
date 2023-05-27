#pragma once

#include "Util.h"

#include <immintrin.h>
#include <intrin.h>
#include <vector>

__forceinline uint64_t BitsCount(uint64_t val) {
    return __popcnt64(val);
}

template<typename F>
__forceinline void ScanBits(uint64_t val, uint64_t baseIndex, F func) {
    unsigned long bitIndex;
    while (_BitScanForward64(&bitIndex, val)) {
        func(baseIndex | bitIndex);
        //val &= ~(1ui64 << bitIndex);
        val = _blsr_u64(val);
    }
}

template<typename F>
__forceinline void ScanTwoBits(uint64_t val, uint64_t baseIndex, F func) {
    unsigned long bitIndex;
    while (_BitScanForward64(&bitIndex, val)) {
        auto pos = bitIndex / 2;
        auto offset = pos * 2;
        func(baseIndex | pos, (val >> offset) & 3);
        val &= ~(3ui64 << offset);
    }
}

template<typename F>
__forceinline void ScanFourBits(uint64_t val, uint64_t baseIndex, F func) {
    unsigned long bitIndex;
    while (_BitScanForward64(&bitIndex, val)) {
        auto pos = bitIndex / 4;
        auto offset = pos * 4;
        func(baseIndex | pos, (val >> offset) & 15);
        val &= ~(15ui64 << offset);
    }
}

class BoolArray {
public:
    BoolArray() {}

    BoolArray(uint64_t size) {
        Resize(size);
    }

    void Resize(uint64_t size) {
        m_Values.resize((size + 63) / 64, 0);
    }

    void Set(uint64_t index) {
        //_bittestandset64((int64_t*)&m_Values[index / 64], index & 63);
        m_Values[index / 64] |= (1ui64 << (index & 63));
    }

    void Clear(uint64_t index) {
        m_Values[index / 64] &= ~(1ui64 << (index & 63));
    }

    bool Get(uint64_t index) {
        return m_Values[index / 64] & (1ui64 << (index & 63));
    }

    void AndNot(const BoolArray& exclude) {
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            m_Values[i] &= ~exclude.m_Values[i];
        }
    }

    void Or(const BoolArray& other) {
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            m_Values[i] |= other.m_Values[i];
        }
    }

    uint64_t BitsCount() {
        uint64_t result = 0;
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            result += ::BitsCount(m_Values[i]);
        }
        return result;
    }

    uint64_t AndNotAndCount(const BoolArray& exclude) {
        uint64_t result = 0;
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            uint64_t val = m_Values[i] & ~exclude.m_Values[i];
            m_Values[i] = val;
            result += ::BitsCount(val);
        }
        return result;
    }

    std::vector<uint64_t>& Data() { return m_Values; }
    const std::vector<uint64_t>& Data() const { return m_Values; }

    void Clear() {
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            m_Values[i] = 0;
        }
    }

    template<typename F>
    void ScanBits(F func) {
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            auto val = m_Values[i];
            if (val == 0) continue;
            ::ScanBits(val, i * 64, func);
        }
    }

    template<typename F>
    void ScanBitsAndClear(F func) {
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            auto val = m_Values[i];
            m_Values[i] = 0;
            if (val == 0) continue;
            ::ScanBits(val, i * 64, func);
        }
    }

private:
    std::vector<uint64_t> m_Values;
};
