// What the loader returns. It never throws.
//
// Flight code compiles `-fno-exceptions` (F' CPP-25), so throwing is impossible
// by construction rather than by discipline. The values are shared with the
// Python `sentinel_export.Status`, so one test asserts both sides refuse the same
// bytes for the same reason. `docs/MODEL_FILE.md` 8 is normative.
#ifndef SENTINEL_STATUS_HPP
#define SENTINEL_STATUS_HPP

#include "sentinel/Types.hpp"

namespace Sentinel {

enum class LoadStatus : U8 {
    OK = 0U,
    BAD_MAGIC = 1U,
    BAD_VERSION = 2U,
    BAD_HEADER_CRC = 3U,
    BAD_STATIC_CRC = 4U,
    BAD_PARAM_CRC = 5U,
    BAD_ARCH = 6U,
    BAD_GATE_ORDER = 7U,
    BAD_SHAPE = 8U,
    TOO_LARGE = 9U,
    TRUNCATED = 10U,
    BAD_NORM_POLICY = 11U
};

//! Human-readable name, for test output and for work item 9's event text.
const char* statusName(LoadStatus status);

}  // namespace Sentinel

#endif  // SENTINEL_STATUS_HPP
