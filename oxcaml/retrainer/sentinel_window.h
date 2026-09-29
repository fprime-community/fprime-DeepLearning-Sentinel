/* D85: the C surface over 56's healthy-window rule (window56.ml, window_c.ml).
 *
 * Every entry point returns int32_t and takes fixed-size integers only (CPP-3,
 * CPP-25). The rule: G-limit (no tick in the trailing W outside any band), G-rate
 * (emissions in W at most k x the model's nominal rate x W) and G-suff (W full),
 * W = 6,550 and k = 2 (docs/MODELS.md 56).
 */
#ifndef SENTINEL_WINDOW_H
#define SENTINEL_WINDOW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SENTINEL_WIN_OK              0
#define SENTINEL_WIN_ERR_ARG        (-1)
#define SENTINEL_WIN_ERR_INTERNAL   (-6)
#define SENTINEL_WIN_ERR_NO_RUNTIME (-7)

/* Sets G-rate's nominal rate from the flying model's own calibrated figure, in
   parts per million. Once, before the first push. */
int32_t sentinel_window_configure(int32_t nominal_ppm);

/* One tick: `limit_any` is 1 if any channel was outside any band, `emitted` is 1
   if the detector emitted. Each is 0 or 1; anything else is SENTINEL_WIN_ERR_ARG. */
int32_t sentinel_window_push(int32_t limit_any, int32_t emitted);

/* 1 if the trailing window is admitted, 0 if refused, negative on failure. */
int32_t sentinel_window_admits(void);

/* Emissions and limit ticks in the current window, for telemetry. */
int32_t sentinel_window_emits(void);
int32_t sentinel_window_limits(void);

int32_t sentinel_window_reset(void);

#ifdef __cplusplus
}
#endif

#endif
