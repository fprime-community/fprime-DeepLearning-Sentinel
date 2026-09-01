// F' primitive type names, so work item 9 swaps this header and nothing else.
//
// F' CPP-3 forbids bare `int`, `float` and `double` in flight code: numerical
// fields use the fixed-size names below, or the configurable `Fw*` types. Work
// item 8 is freestanding (D31), so the names are defined here from <cstdint>
// with exactly the meanings `nasa/fprime` v4.3.0 `docs/reference/numerical-types.md`
// gives them. Work item 9 replaces this file's body with
// `#include "Fw/FPrimeBasicTypes.hpp"` and no other file changes.
#ifndef SENTINEL_TYPES_HPP
#define SENTINEL_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace Sentinel {

using U8 = std::uint8_t;
using U16 = std::uint16_t;
using U32 = std::uint32_t;
using U64 = std::uint64_t;
using I32 = std::int32_t;

// F32 is `float` and F64 is `double`, as F' names them.
//
// F' treats F64 as a *configurable* platform feature (`FW_HAS_F64` in
// `PlatformTypes.h`) that a target may switch off. Sentinel requires it: the
// frozen decision layer is float64 by construction -- `aggregate_predictions`
// accumulates in float64, `telemanom.ewma` computes in float64, and the
// threshold comparison at `harness.py:169-172` is float64. Phase 4's board
// selection inherits that constraint. See `docs/MODEL_FILE.md` 9.
using F32 = float;
using F64 = double;

//: Sizes and counts. `FwSizeType` in an F' build.
using SizeType = std::size_t;

}  // namespace Sentinel

#endif  // SENTINEL_TYPES_HPP
