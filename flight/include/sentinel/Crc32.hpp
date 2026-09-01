// CRC-32/IEEE 802.3, the same one F' computes.
//
// Reversed polynomial 0xEDB88320, initial value 0xFFFFFFFF, final complement.
// Check value: crc32("123456789") == 0xCBF43926.
//
// This is exactly `zlib.crc32` on the Python side and exactly F's `Utils::Hash`
// on the flight side -- `nasa/fprime` v4.3.0 `Utils/Hash/Crc32/Crc32.hpp`
// implements the same polynomial and applies the one's complement in the `Hash`
// class. Work item 8 is freestanding (D31) so the table is carried here; work
// item 9 may substitute `Utils::Hash` and the value will not change.
#ifndef SENTINEL_CRC32_HPP
#define SENTINEL_CRC32_HPP

#include "sentinel/Types.hpp"

namespace Sentinel {

//! Streaming state, so a CRC accumulates as bytes are consumed and the file is
//! never buffered whole (`docs/MODEL_FILE.md` 8).
class Crc32 {
  public:
    Crc32() : m_state(0xFFFFFFFFU) {}

    void reset() { m_state = 0xFFFFFFFFU; }
    void update(const U8* data, U32 length);
    U32 value() const { return m_state ^ 0xFFFFFFFFU; }

  private:
    U32 m_state;
};

//! One-shot, for a buffer already to hand.
U32 crc32(const U8* data, U32 length);

}  // namespace Sentinel

#endif  // SENTINEL_CRC32_HPP
