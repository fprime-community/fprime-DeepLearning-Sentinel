// Compile-time bounds. Every buffer in the core is sized from these.
//
// F' CPP-1 forbids allocation after initialisation and permits "pre-sized arrays
// sized at compile or init time"; this core uses the compile-time half and has no
// allocator at all. The work item's phrasing -- "buffers sized at compile time
// from the model header" -- cannot be done as written, because a header is read at
// runtime. What happens instead: buffers are sized from the constants below, and
// `ModelFile` validates the header against them, refusing a larger model with
// `TOO_LARGE` rather than reading past the end of anything.
//
// F' CPP-8 and CPP-30: typed constants with their derivation, never bare literals.
#ifndef SENTINEL_CONFIG_HPP
#define SENTINEL_CONFIG_HPP

#include "sentinel/Types.hpp"

namespace Sentinel {
namespace Config {

// The flown configuration is 12 channels; 16 leaves room for a wider instance
// without touching the code. Objective.md 10.2 fix 2 models 8-12 channels in one
// subsystem and scales by adding instances, not by widening one model.
constexpr U32 MAX_CHANNELS = 16U;

// Two layers of 80 is the architecture the gate selected (D28). The file format
// carries four hidden slots so it can describe a deeper model; this build budgets
// two and refuses more, which is what TOO_LARGE exists to demonstrate.
constexpr U32 MAX_LAYERS = 2U;
constexpr U32 MAX_HIDDEN = 80U;

// telemanom's l_p. The head emits this many future steps for every channel.
constexpr U32 MAX_PREDICTIONS = 10U;

// n_channels + n_exogenous. n_exogenous is 0 until D6's command inputs are
// measured, so this equals MAX_CHANNELS today.
constexpr U32 MAX_INPUTS = MAX_CHANNELS;

// The GRU's three gate blocks, stacked along axis 0 of w_ih and w_hh.
constexpr U32 N_GATES = 3U;

// Derived bounds, so no buffer declaration carries arithmetic of its own.
constexpr U32 MAX_GATE_WIDTH = N_GATES * MAX_HIDDEN;          // 240
constexpr U32 MAX_OUTPUTS = MAX_PREDICTIONS * MAX_CHANNELS;   // 160

// Largest w_ih is layer 0's (MAX_GATE_WIDTH x MAX_INPUTS) or a later layer's
// (MAX_GATE_WIDTH x MAX_HIDDEN); take the wider input.
constexpr U32 MAX_W_IH_COLUMNS =
    (MAX_INPUTS > MAX_HIDDEN) ? MAX_INPUTS : MAX_HIDDEN;
constexpr U32 MAX_W_IH = MAX_GATE_WIDTH * MAX_W_IH_COLUMNS;
constexpr U32 MAX_W_HH = MAX_GATE_WIDTH * MAX_HIDDEN;
constexpr U32 MAX_HEAD_W = MAX_OUTPUTS * MAX_HIDDEN;

// 16-byte NUL-padded ASCII, as `docs/MODEL_FILE.md` 4 specifies.
constexpr U32 CHANNEL_NAME_BYTES = 16U;

}  // namespace Config
}  // namespace Sentinel

#endif  // SENTINEL_CONFIG_HPP
