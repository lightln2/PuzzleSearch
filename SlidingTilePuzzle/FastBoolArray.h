#pragma once

#include "Util.h"

#include <immintrin.h>
#include <vector>

class FastBitArray {
public:
    void Resize(uint64_t size) {
        m_Values.resize((size + 63) / 64, 0);
    }

    void Set(uint64_t index) {
        m_Values[index / 64] |= (1ui64 << (index & 63));
    }

    template<typename F>
    void ScanBits(F func) {
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            auto val = m_Values[i];
            unsigned long bitIndex;
            while (_BitScanForward64(&bitIndex, val)) {
                func((i * 64) | bitIndex);
                //val = _blsr_u64(val);
                val &= ~(1ui64 << bitIndex);
            }
        }
    }

    int64_t NextBit() {
        if (m_ReaderPos == 0) {
            NextWord();
        }
        unsigned long bitIndex;
        if (_BitScanForward64(&bitIndex, m_CurrentValue)) {
            m_CurrentValue &= ~(1ui64 << bitIndex);
            int64_t fullIndex = (m_ReaderPos - 1) * 64 + bitIndex;
            return fullIndex;
        }
        else {
            m_Values[m_ReaderPos - 1] = 0;
            NextWord();
            if (m_ReaderPos == m_Values.size() && m_CurrentValue == 0) {
                m_ReaderPos = 0;
                return -1;
            }
        }
    }

private:
    void NextWord() {
        while (m_CurrentValue == 0 && m_ReaderPos < m_Values.size()) {
            m_CurrentValue = m_Values[m_ReaderPos++];
        }
    }
private:
    std::vector<uint64_t> m_Values;

    size_t m_ReaderPos = 0;
    uint64_t m_CurrentValue = 0;
};
