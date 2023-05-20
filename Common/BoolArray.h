#pragma once

#include "Util.h"

#include <immintrin.h>
#include <intrin.h>
#include <vector>

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

    bool Get(uint64_t index) {
        return m_Values[index / 64] & (1ui64 << (index & 63));
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
            unsigned long bitIndex;
            while (_BitScanForward64(&bitIndex, val)) {
                func((i * 64) | bitIndex);
                //val &= ~(1ui64 << bitIndex);
                val = _blsr_u64(val);
            }
        }
    }

    template<typename F>
    void ScanBitsAndClear(F func) {
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            auto val = m_Values[i];
            m_Values[i] = 0;
            if (val == 0) continue;
            unsigned long bitIndex;
            while (_BitScanForward64(&bitIndex, val)) {
                func((i * 64) | bitIndex);
                //val &= ~(1ui64 << bitIndex);
                val = _blsr_u64(val);
            }
        }
    }

private:
    std::vector<uint64_t> m_Values;
};
