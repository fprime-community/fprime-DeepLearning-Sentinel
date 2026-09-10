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

// -- telemanom's dynamic threshold, and the derivative stream (docs/MODELS.md 39)
//
// (!) THESE ARE `constexpr` AND NOT `model.bin` FIELDS, BY DECISION. 39.3:
// `docs/MODEL_FILE.md` 11 makes any new field a `format_version` bump and refuses
// a non-zero reserved field, and a selectable rule stays a version-2 decision
// taken after Phase 3 (D62 consequence 2). PARAMS already carries the two the
// format did anticipate -- `ewma_span` 105 and `warmup_steps` 2,350 -- and those
// are read from the file, not from here.
//
// Every value is transcribed from `src/sentinel_models/telemanom.py` at the line
// beside it, read at first hand and never recalled.

// `telemanom.py:84-85`. batch_size 70 x window_size 30. The error window the
// threshold is measured over, and the span the derivative is standardised on
// (`scripts/decision_layer_arms.py:224` uses `span = error_window`).
constexpr U32 ERROR_WINDOW_BATCH = 70U;
constexpr U32 ERROR_WINDOW_COUNT = 30U;
constexpr U32 ERROR_WINDOW = ERROR_WINDOW_BATCH * ERROR_WINDOW_COUNT;   // 2100

// `telemanom.py:374-450`, `channel_ratios`: the threshold is re-solved every
// `stride` steps and applied to those steps. Equal to the batch, which is what
// makes a window 2,170 samples and 39.4's guard-cell question a real one.
constexpr U32 STRIDE = ERROR_WINDOW_BATCH;

// `telemanom.py:100-102`. The published sweep, `np.arange(2.5, 12, 0.5)`: 19
// candidates. A default of the method rather than a constant of it -- the
// docstring there records why -- but it is what the reference sweeps and what
// this transcribes.
constexpr F32 Z_FLOOR = 2.5F;
constexpr F32 Z_CEILING = 12.0F;
constexpr F32 Z_STEP = 0.5F;
constexpr U32 Z_CANDIDATES = 19U;   // floor + n*step < ceiling, n = 0..18

// `telemanom.py:86`. Each exceedance run is widened by ERROR_BUFFER - 1 = 99.
// (!) FORWARD ONLY IN FLIGHT (39.5): the backward half would mark timesteps
// already emitted, and there is no un-emit. The price is registered as N6 and is
// NOT YET MEASURED.
constexpr U32 ERROR_BUFFER = 100U;

// `telemanom.py:87`. The pruning ladder's relative-drop floor.
constexpr F32 PRUNING_P = 0.13F;

// The largest number of exceedance runs a single window can hold: a run needs at
// least one sample above and one below to be distinguishable, so the bound is
// half the window, and CPP-34 wants every loop bounded by a compile-time value.
constexpr U32 MAX_SEQUENCES = ERROR_WINDOW / 2U;

// The Level 1 statistical baseline's trailing window, from `RollingStd`'s only
// ever-used value -- `src/sentinel_models/baselines.py:91`, `window = 120`, which
// the registry never overrides (`registry.py:38`). It is also the baseline's
// warm-up, `baselines.py:96-98`: 120 ticks, against the model path's 2,350.
constexpr U32 BASELINE_WINDOW = 120U;

}  // namespace Config
}  // namespace Sentinel

#endif  // SENTINEL_CONFIG_HPP
