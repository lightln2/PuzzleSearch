#pragma once

#include "Buffer.h"
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

template<typename F>
__forceinline void ScanBits(uint64_t val, uint64_t baseIndex, int bits, F func) {
    unsigned long bitIndex;
    uint64_t mask = (1ui64 << bits) - 1;
    while (_BitScanForward64(&bitIndex, val)) {
        auto pos = bitIndex / bits;
        auto offset = pos * bits;
        func(baseIndex | pos, int((val >> offset) & mask));
        val &= ~(mask << offset);
    }
}

template<int BITS, typename F>
__forceinline void ScanNBits(uint64_t val, uint64_t baseIndex, F func) {
    static constexpr uint64_t MASK = (1ui64 << BITS) - 1;
    unsigned long bitIndex;
    while (_BitScanForward64(&bitIndex, val)) {
        auto pos = bitIndex / BITS;
        auto offset = pos * BITS;
        func(baseIndex | pos, int((val >> offset) & MASK));
        val &= ~(MASK << offset);
    }
}

class BitArray {
public:
    BitArray() {}

    BitArray(uint64_t size)
        : m_Values((size + 63) / 64)
    { }

    void Resize(uint64_t size) {
        m_Values.resize((size + 63) / 64);
    }

    void Set(uint64_t index) {
        m_Values[index / 64] |= (1ui64 << (index & 63));
    }

    void Clear(uint64_t index) {
        m_Values[index / 64] &= ~(1ui64 << (index & 63));
    }

    bool Get(uint64_t index) {
        return m_Values[index / 64] & (1ui64 << (index & 63));
    }

    void AndNot(const BitArray& exclude) {
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            m_Values[i] &= ~exclude.m_Values[i];
        }
    }

    void Or(const BitArray& other) {
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

    uint64_t AndNotAndCount(const BitArray& exclude) {
        uint64_t result = 0;
        for (uint64_t i = 0; i < m_Values.size(); i++) {
            uint64_t val = m_Values[i] & ~exclude.m_Values[i];
            m_Values[i] = val;
            result += ::BitsCount(val);
        }
        return result;
    }

    size_t DataSize() const { return m_Values.size(); }

    uint64_t* Data() { return &m_Values[0]; }

    const uint64_t* Data() const { return &m_Values[0]; }

    void Clear() {
        for (size_t i = 0; i < m_Values.size(); i++) m_Values[i] = 0;
    }

    template<typename F>
    void ScanBits(F func) {
        for (size_t i = 0; i < m_Values.size(); ++i) {
            const uint64_t val = m_Values[i];
            if (val == 0) continue;
            ::ScanBits(val, i * 64, func);
        }
    }

    template<typename F>
    void ScanBitsAndClear(F func) {
        for (size_t i = 0; i < m_Values.size(); ++i) {
            const uint64_t val = m_Values[i];
            if (val == 0) continue;
            m_Values[i] = 0;
            ::ScanBits(val, i * 64, func);
        }
    }

    template<typename F>
    void ScanBitsAndClearWithExcl(F func, const BitArray& excl) {
        const auto* exclData = excl.Data();
        for (size_t i = 0; i < m_Values.size(); ++i) {
            const uint64_t val = m_Values[i];
            if (val == 0) continue;
            m_Values[i] = 0;
            ::ScanBits(val & ~exclData[i], i * 64, func);
        }
    }

private:
    std::vector<uint64_t> m_Values;
};

template <int BITS>
class MultiBitArray {
    static_assert(BITS > 0 && BITS <= 16);
    static_assert((BITS & (BITS - 1)) == 0);
    static constexpr int VALS_PER_WORD = 64 / BITS;

public:
    MultiBitArray(uint64_t size)
        : m_Array(size * BITS)
    {}

    void Set(uint64_t index, int bit) {
        m_Array.Set((index * BITS) | bit);
    }

    uint64_t* Data() { return m_Array.Data(); }
    const uint64_t* Data() const { return m_Array.Data(); }
    size_t DataSize() const { return m_Array.DataSize(); }

    template<typename F>
    void ScanBitsAndClear(F func) {
        for (uint64_t i = 0; i < m_Array.DataSize(); i++) {
            auto val = m_Array.Data()[i];
            m_Array.Data()[i] = 0;
            if (val == 0) continue;
            ::ScanNBits<BITS>(val, i * VALS_PER_WORD, func);
        }
    }

private:
    BitArray m_Array;
};

class IndexedBitArray {
    static constexpr size_t STEP = 1024;
public:
    IndexedBitArray(uint64_t size)
        : m_Array(size)
        , m_Index((m_Array.DataSize() + STEP - 1) / STEP)
    {}

    void Set(uint64_t index) {
        m_Array.Set(index);
        m_Index.Set((index) / (64 * STEP));
    }

    bool Get(uint64_t index) {
        return m_Array.Get(index);
    }

    void Clear() {
        m_Array.Clear();
        m_Index.Clear();
    }

    void Clear(uint64_t index) {
        m_Array.Clear(index);
    }

    uint64_t* Data() { return m_Array.Data(); }
    const uint64_t* Data() const { return m_Array.Data(); }
    size_t DataSize() const { return m_Array.DataSize(); }

    template<typename F>
    void ScanBitsAndClear(F func) {
        uint64_t* ptr = Data();
        uint64_t arrSize = DataSize();
        m_Index.ScanBitsAndClear([&](uint64_t index) {
            uint64_t offset = index * STEP;
            uint64_t end = std::min(arrSize, offset + STEP);
            for (size_t i = offset; i < end; i++) {
                uint64_t val = ptr[i];
                if (val == 0) continue;
                ptr[i] = 0;
                ::ScanBits(val, i * 64, func);
            }
        });
    }

    template<typename F>
    void ScanBitsAndClearWithExcl(F func, const BitArray& exclude) {
        uint64_t* ptr = Data();
        uint64_t arrSize = DataSize();
        const uint64_t* exclPtr = exclude.Data();
        m_Index.ScanBitsAndClear([&](uint64_t index) {
            uint64_t offset = index * STEP;
            uint64_t end = std::min(arrSize, offset + STEP);
            for (size_t i = offset; i < end; i++) {
                uint64_t val = ptr[i];
                if (val == 0) continue;
                ptr[i] = 0;
                ::ScanBits(val & ~exclPtr[i], i * 64, func);
            }
        });
    }

private:
    BitArray m_Array;
    BitArray m_Index;
};

template<int BITS>
class IndexedArray {
    static constexpr size_t STEP = 1024;
public:
    IndexedArray(uint64_t size)
        : m_Array(size)
        , m_Index((m_Array.DataSize() + STEP - 1)/ STEP)
    {}

    void Set(uint64_t index, int bit) {
        m_Array.Set(index, bit);
        m_Index.Set((index * BITS) / (64 * STEP));
    }

    uint64_t* Data() { return m_Array.Data(); }
    const uint64_t* Data() const { return m_Array.Data(); }
    size_t DataSize() const { return m_Array.DataSize(); }

    template<typename F>
    void ScanBitsAndClear(F func) {
        uint64_t* ptr = Data();
        uint64_t arrSize = DataSize();
        m_Index.ScanBitsAndClear([&](uint64_t index) {
            uint64_t offset = index * STEP;
            uint64_t end = std::min(arrSize, offset + STEP);
            for (size_t i = offset; i < end; i++) {
                uint64_t val = ptr[i];
                if (val == 0) continue;
                ptr[i] = 0;
                ::ScanNBits<BITS>(val, i * (64 / BITS), func);
            }
        });
    }

private:
    MultiBitArray<BITS> m_Array;
    BitArray m_Index;
};
