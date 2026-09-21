/* Section 61 / E5-c: the C surface over the float32 training cycle.
 *
 * (!) EVERY ENTRY POINT RETURNS int32_t AND TAKES FIXED-SIZE TYPES ONLY.
 * CPP-3 (SKILL.md:114-124): fixed-size numerical types. CPP-21 (:151-156): an
 * array crossing is always a pointer and a length, never a bare array.
 * CPP-25 (:279-293): no exceptions -- a refusal is a status code.
 *
 * (!) THE BUFFERS ARE THE CALLER'S. sentinel_cycle_load and sentinel_cycle_export
 * wrap the caller's memory with CAML_BA_EXTERNAL and the wrapper does not outlive
 * the call. Nothing here allocates after init, which is CPP-1 (:41-50).
 */
#ifndef SENTINEL_CYCLE_H
#define SENTINEL_CYCLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SENTINEL_CYC_OK              0
#define SENTINEL_CYC_ERR_SHAPE      (-1)
#define SENTINEL_CYC_ERR_BUDGET     (-2)
#define SENTINEL_CYC_ERR_NO_RUNTIME (-7)

/* Boots the OCaml runtime. Idempotent; caml_startup runs at most once per
   process, which is why one retrainer per process is a design requirement
   (D70 consequence 9). */
int32_t sentinel_cycle_boot(void);

/* Seeds a reproducible starting model. */
int32_t sentinel_cycle_init(int32_t salt);

/* Copies the caller's window in. `n` is in float32 elements. */
int32_t sentinel_cycle_load(const float *window, uint32_t n);

/* Runs one retraining cycle: exactly `budget` optimiser steps (D73), over
   `t_steps` timesteps, or none at all if `admit` is zero (56's window gate). */
int32_t sentinel_cycle_run(int32_t budget, int32_t t_steps, int32_t admit);

/* How many steps the last cycle took. Negative on failure. */
int32_t sentinel_cycle_steps(void);

/* Copies the best-weights block out into the caller's buffer. */
int32_t sentinel_cycle_export(float *out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif
