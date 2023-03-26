#pragma once

#include "File.h"

#include <cstdint>

class StreamVInt {
public:
    static constexpr int MAX_INDEXES_COUNT = 256 * 1024;
    static constexpr int MAX_BUFFER_SIZE = 1 * 1024 * 1024;

public:
    // for testing
    static int EncodeTuple(const uint32_t* indexes, uint8_t* buffer);
    static int DecodeTuple(const uint8_t* buffer, uint32_t* indexes);
    static void EncodeBoundsTuple(const uint8_t* bounds, uint8_t* buffer);
    static void DecodeBoundsTuple(const uint8_t* buffer, uint8_t* bounds);

    static int EncodeIndexes(int count, const uint32_t* indexes, uint8_t* buffer);
    static int DecodeIndexes(int count, const uint8_t* buffer, uint32_t* indexes);
    // TODO: investigate why decode is slower than encode
    static int DecodeIndexesAndDiff(int count, const uint8_t* buffer, uint32_t* indexes);

    static int EncodeBounds(int count, const uint8_t* bounds, uint8_t* buffer);
    static int DecodeBounds(int count, const uint8_t* buffer, uint8_t* bounds);

    static int Encode(int count, uint32_t* indexes, uint8_t* buffer, int buffer_capacity);
    static int Decode(int& size, uint8_t* buffer, uint32_t* indexes, int values_capacity);
    static int Encode(int count, uint32_t* indexes, uint8_t* bounds, uint8_t* buffer, int buffer_capacity);
    static int Decode(int& size, uint8_t* buffer, uint32_t* indexes, uint8_t* bounds, int values_capacity);

    static void Encode(Buffer<uint32_t>& indexes, Buffer<uint8_t>& buffer) {
        size_t encoded = Encode(
            indexes.Size(),
            indexes.Buf(),
            buffer.Buf() + buffer.Size(),
            buffer.Capacity() - buffer.Size());
        buffer.SetSize(buffer.Size() + encoded);
    }

    static int Decode(int position, Buffer<uint8_t>& buffer, Buffer<uint32_t>& indexes) {
        int size = buffer.Size() - position;
        size_t decoded = Decode(size, buffer.Buf() + position, indexes.Buf(), indexes.Capacity());
        indexes.SetSize(decoded);
        return position + size;
    }

    static void Encode(Buffer<uint32_t>& indexes, Buffer<uint8_t>& bounds, Buffer<uint8_t>& buffer) {
        int encoded = Encode(
            indexes.Size(),
            indexes.Buf(),
            bounds.Buf(),
            buffer.Buf() + buffer.Size(),
            buffer.Capacity() - buffer.Size());
        buffer.SetSize(buffer.Size() + encoded);
    }

    static int Decode(int position, Buffer<uint8_t>& buffer, Buffer<uint32_t>& indexes, Buffer<uint8_t>& bounds) {
        int size = buffer.Size() - position;
        size_t decoded = Decode(size, buffer.Buf() + position, indexes.Buf(), bounds.Buf(), indexes.Capacity());
        indexes.SetSize(decoded);
        return position + size;
    }

    static void PrintStats();
private:
    static std::atomic<uint64_t> m_StatEncodeNanos;
    static std::atomic<uint64_t> m_StatDecodeNanos;
};

