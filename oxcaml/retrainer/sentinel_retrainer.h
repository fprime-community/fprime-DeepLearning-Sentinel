/* E1, the C face of the OxCaml retrainer. docs/MODELS.md 47.6, boundary B.
 *
 * This header is the ENTIRE surface the C++ side may touch. Everything crossing is a
 * fixed-size scalar or a pointer-plus-length pair; no OCaml value appears here, and none
 * persists in C++ across a call (docs/MODELS.md 47.9, prediction X3).
 *
 * F' rules this is written to, quoted at docs/PHASE5.md 3 from
 * .github/skills/fprime-cpp-design/SKILL.md at nasa/fprime v4.3.0:
 *   CPP-3   fixed-size numerical types. int32_t/uint32_t/double, never int or float.
 *   CPP-21  no C-style arrays in interfaces; every array is paired with its length.
 *   CPP-25  no exceptions. Every OCaml exception is caught inside OCaml and returned
 *           as a status code, so nothing can propagate into -fno-exceptions C++.
 *
 * (!) THE RUNTIME MUST BE BOOTED BEFORE ANY OTHER CALL. sentinel_retrainer_boot()
 * starts the OCaml runtime; calling anything else first is undefined.
 */
#ifndef SENTINEL_RETRAINER_H
#define SENTINEL_RETRAINER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. Mirror retrainer.ml's constants exactly. */
#define SENTINEL_RT_OK               0
#define SENTINEL_RT_ERR_NOT_INIT     1
#define SENTINEL_RT_ERR_ALREADY_INIT 2
#define SENTINEL_RT_ERR_CAPACITY     3
#define SENTINEL_RT_ERR_OVERFLOW     4
#define SENTINEL_RT_ERR_SHORT_BUFFER 5
#define SENTINEL_RT_ERR_INTERNAL     6
#define SENTINEL_RT_ERR_NO_RUNTIME   7  /* C-side only: boot() not called, or failed */

/* State codes returned by sentinel_retrainer_status(). */
#define SENTINEL_RT_STATE_UNINIT   0
#define SENTINEL_RT_STATE_READY    1
#define SENTINEL_RT_STATE_FED      2
#define SENTINEL_RT_STATE_STEPPED  3

/* Boots the OCaml runtime. Idempotent. Returns SENTINEL_RT_OK or
 * SENTINEL_RT_ERR_NO_RUNTIME. Must be called once before anything below. */
int32_t sentinel_retrainer_boot(void);

/* The five entry points E1 registers (docs/MODELS.md 47.9, X1). */
int32_t sentinel_retrainer_init(uint32_t capacity);
int32_t sentinel_retrainer_feed(const double *samples, uint32_t n);
int32_t sentinel_retrainer_step(void);
int32_t sentinel_retrainer_status(void);
int32_t sentinel_retrainer_export(double *out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* SENTINEL_RETRAINER_H */
