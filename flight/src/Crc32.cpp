#include "sentinel/Crc32.hpp"

namespace Sentinel {
namespace {

//! Built once at load, not typed out: a 256-entry literal table is 256 chances to
//! transpose a digit, and F' CPP-30 asks that a constant show its derivation.
//! The check value in `flight/test/RefusalTests.cpp` proves the table is right.
struct Table {
    U32 entry[256];

    Table() : entry() {
        for (U32 i = 0U; i < 256U; ++i) {
            U32 value = i;
            for (U32 bit = 0U; bit < 8U; ++bit) {
                value = ((value & 1U) != 0U) ? (0xEDB88320U ^ (value >> 1U))
                                             : (value >> 1U);
            }
            entry[i] = value;
        }
    }
};

const Table& table() {
    // Function-local static: initialised once, before any use, with no allocation
    // and no static-initialisation-order dependency.
    static const Table built;
    return built;
}

}  // namespace

void Crc32::update(const U8* data, U32 length) {
    if (data == nullptr) {
        return;
    }
    const Table& lookup = table();
    U32 state = m_state;
    for (U32 i = 0U; i < length; ++i) {
        const U32 index = (state ^ static_cast<U32>(data[i])) & 0xFFU;
        state = lookup.entry[index] ^ (state >> 8U);
    }
    m_state = state;
}

U32 crc32(const U8* data, U32 length) {
    Crc32 accumulator;
    accumulator.update(data, length);
    return accumulator.value();
}

}  // namespace Sentinel
