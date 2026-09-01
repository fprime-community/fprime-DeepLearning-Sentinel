// F' primitive type names, so work item 9 swaps this header and nothing else.
//
// F' CPP-3 forbids bare `int`, `float` and `double` in flight code: numerical
// fields use the fixed-size names below, or the configurable `Fw*` types. The
// names carry exactly the meanings `nasa/fprime` v4.3.0
// `docs/reference/numerical-types.md` gives them.
//
// TWO WORLDS, ONE CORE (D35). Work item 8 built this core freestanding so the
// arithmetic could be verified without a framework, which is D15's premise. Work
// item 9 puts the same sources under F'. D31 consequence 2 said work item 9
// would "replace that one header with `Fw/FPrimeBasicTypes.hpp`", and that is
// not achievable as written: `Fw/FPrimeBasicTypes.hpp` includes
// `config/FppConstantsAc.hpp`, which the F' build generates and which does not
// exist for `make -C flight test`. So the header selects instead of replacing.
// The claim D31 was protecting still holds exactly: this file is the ONLY file
// that differs between the two builds, and `tests/test_flight_build.py` asserts
// it.
//
// `SENTINEL_FPRIME_TYPES` is defined by the F' build alone.
#ifndef SENTINEL_TYPES_HPP
#define SENTINEL_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(SENTINEL_FPRIME_TYPES)
#include "Fw/FPrimeBasicTypes.hpp"
#endif

namespace Sentinel {

#if defined(SENTINEL_FPRIME_TYPES)

// F' declares these at global scope in `Fw/Types/BasicTypes.h`. Aliasing rather
// than redeclaring them is what keeps every other file in this directory
// byte-identical between the two builds.
using U8 = ::U8;
using U16 = ::U16;
using U32 = ::U32;
using U64 = ::U64;
using I32 = ::I32;
using F32 = ::F32;
using F64 = ::F64;

#else

using U8 = std::uint8_t;
using U16 = std::uint16_t;
using U32 = std::uint32_t;
using U64 = std::uint64_t;
using I32 = std::int32_t;

// F32 is `float` and F64 is `double`, as F' names them.
using F32 = float;
using F64 = double;

#endif

//: Sizes and counts. Identical in both builds; F' does not redefine `size_t`.
using SizeType = std::size_t;

// The dtype map in `docs/MODEL_FILE.md` 9 is not uniform and must not be made
// uniform: the decision layer accumulates in F64 and narrows once, and a core
// that ran it in F32 throughout would be a different detector. These assert the
// property that requirement actually needs, in both builds.
//
// (!) They do NOT test `FW_HAS_F64`. `docs/MODEL_FILE.md` 9 and this header's
// previous text both said F' "treats F64 as a configurable platform type that a
// platform may switch off (`FW_HAS_F64` in `PlatformTypes.h`)". That is not true
// at v4.3.0 and was not true at v4.2.2. `F64` is defined unconditionally at
// `Fw/Types/BasicTypes.h:86` -- "Required for compiler-supplied double
// promotion" -- and `FW_HAS_F64` appears exactly once in the whole framework, in
// the documentation table at `docs/reference/numerical-types.md:35`, which is
// the table work item 8 read. F' documents a macro its code does not define.
// The guarded feature is `FW_HAS_64_BIT`, and it guards `U64`/`I64`, not `F64`.
// So the constraint is asserted here directly rather than through a macro that
// does not exist. See `docs/MODELS.md` 20.2 correction 13.
static_assert(sizeof(F32) == 4U, "F32 must be 4 bytes; the golden vectors are F32");
static_assert(sizeof(F64) == 8U, "F64 must be 8 bytes; the decision layer is F64");
static_assert(std::numeric_limits<F64>::is_iec559,
              "F64 must be IEEE 754; equivalence with the NumPy reference assumes it");
static_assert(std::numeric_limits<F32>::is_iec559,
              "F32 must be IEEE 754; equivalence with the NumPy reference assumes it");

}  // namespace Sentinel

#endif  // SENTINEL_TYPES_HPP
