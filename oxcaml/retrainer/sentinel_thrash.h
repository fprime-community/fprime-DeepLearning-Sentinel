/* E2 apparatus: the GC under deliberate abuse. docs/MODELS.md 47.9.1.
 * Not part of the retrainer's interface; nothing here would exist in a flight build. */
#ifndef SENTINEL_THRASH_H
#define SENTINEL_THRASH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t sentinel_retrainer_boot(void);          /* from sentinel_retrainer.h, reused */

int32_t sentinel_thrash_start_domain(void);     /* arm C: a domain in THIS process */
int32_t sentinel_thrash_run_for(int32_t n, int32_t ms, const char *path);
int32_t sentinel_thrash_stop(void);             /* -> the progress counter */
int32_t sentinel_thrash_count(void);

#ifdef __cplusplus
}
#endif

#endif
