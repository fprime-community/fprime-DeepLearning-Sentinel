// Shared scaffolding for the flight test binaries.
//
// Test code, not flight code, but it is held to the same rules: no exceptions,
// no STL containers, no allocation. The buffers below are static and sized for
// the largest file the suite reads (285,136 bytes at the flown shape).
#ifndef SENTINEL_TEST_SUPPORT_HPP
#define SENTINEL_TEST_SUPPORT_HPP

#include <cstdio>
#include <cstring>

#include "sentinel/Types.hpp"

namespace SentinelTest {

using Sentinel::F32;
using Sentinel::F64;
using Sentinel::U32;
using Sentinel::U8;

constexpr U32 MAX_FILE_BYTES = 512U * 1024U;

extern int failures;

//! Read a whole file into `buffer`. Returns bytes read, or 0 on any failure.
inline U32 readFile(const char* path, U8* buffer, U32 capacity) {
    std::FILE* handle = std::fopen(path, "rb");
    if (handle == nullptr) {
        return 0U;
    }
    const size_t read = std::fread(buffer, 1U, static_cast<size_t>(capacity), handle);
    (void)std::fclose(handle);
    return static_cast<U32>(read);
}

inline bool exists(const char* path) {
    std::FILE* handle = std::fopen(path, "rb");
    if (handle == nullptr) {
        return false;
    }
    (void)std::fclose(handle);
    return true;
}

inline void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("    FAIL  %s\n", what);
        failures += 1;
    }
}

inline void checkEqualU32(U32 actual, U32 expected, const char* what) {
    if (actual != expected) {
        std::printf("    FAIL  %s: got %u, expected %u\n", what, actual, expected);
        failures += 1;
    }
}

inline int report(const char* suite) {
    if (failures == 0) {
        std::printf("  %s: all checks passed\n", suite);
        return 0;
    }
    std::printf("  %s: %d FAILURES\n", suite, failures);
    return 1;
}

}  // namespace SentinelTest

#endif  // SENTINEL_TEST_SUPPORT_HPP
