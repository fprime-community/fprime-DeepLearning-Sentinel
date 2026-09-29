/* Section 72 / E5-e: the C surface over the shadow model file.
 *
 * (!) EVERY ENTRY POINT RETURNS int32_t AND TAKES FIXED-SIZE TYPES ONLY.
 * CPP-3 (SKILL.md:114-124): fixed-size numerical types. CPP-21 (:151-156): an
 * array crossing is always a pointer and a length, never a bare array.
 * CPP-25 (:279-293): no exceptions -- a refusal is a status code, and unlike
 * sentinel_cycle.h's surface this one is structurally guarded on the OCaml side
 * (shadow_c.ml:38, retrainer.ml:41-44's pattern).
 *
 * (!) THE BUFFERS ARE THE CALLER'S. Each entry wraps the caller's memory with
 * CAML_BA_EXTERNAL and the wrapper does not outlive the call. Nothing here
 * allocates after init, which is CPP-1 (:41-50).
 *
 * (!) THERE IS NO BOOT OF ITS OWN. One -output-complete-obj object carries one
 * runtime (scripts/oxcaml_s61.sh:79-85), so this surface shares the cycle's.
 * Each entry calls sentinel_cycle_boot(), which is idempotent
 * (cycle_stubs.c:26-28), rather than starting a second one.
 */
#ifndef SENTINEL_SHADOW_H
#define SENTINEL_SHADOW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 0, -1 and -2 are Shadow59.write_shadow's own returns, passed through unchanged
   (shadow59.ml:79-80), so one code means one thing on both sides. */
#define SENTINEL_SHD_OK              0
#define SENTINEL_SHD_ERR_SHAPE      (-1)
#define SENTINEL_SHD_ERR_CAPACITY   (-2)
#define SENTINEL_SHD_ERR_INTERNAL   (-6)
#define SENTINEL_SHD_ERR_NO_RUNTIME (-7)

/* Copies the flying model file's bytes IN, to be overwritten in place.
   `n` is in bytes. */
int32_t sentinel_shadow_load(const uint8_t *flying, uint32_t n);

/* Overwrites the weights payload with the caller's block and patches
   static_crc32 (44) and header_crc32 (60). `n` is in float32 elements.
   D30's freeze is not engaged: no field is added, moved or resized, and
   format_version stays 1 (shadow59.ml:13-15). */
int32_t sentinel_shadow_write(const float *weights, uint32_t n);

/* Copies the candidate OUT into the caller's buffer. `cap` is the buffer's size
   in bytes.

   (!) RETURNS THE BYTE COUNT WRITTEN, POSITIVE, not SENTINEL_SHD_OK. A candidate
   of zero bytes is not a thing, so positive-is-length and negative-is-refusal is
   unambiguous, and the caller never has to assume the candidate is the same
   length as the flying file even though it always is. */
int32_t sentinel_shadow_export(uint8_t *out, uint32_t cap);

/* The last cycle's loss, written into out[0]. `n` is in float32 elements and
   must be at least 1.

   (!) IT IS ON THIS SURFACE AND NOT sentinel_cycle.h's, because HO2's sanity
   report needs it and 61's surface -- a measured module this section does not
   edit -- has no metrics entry. shadow_c.ml carries the reasoning. */
int32_t sentinel_shadow_loss(float *out, uint32_t n);

/* D85 / D76: copies the loaded flying file's WEIGHTS into the training cycle's
   parameters -- the onboard warm start. Call after sentinel_shadow_load. Refuses
   (SENTINEL_SHD_ERR_SHAPE) a weights block that is not this shape's parameter
   count, or a header that disagrees with the bytes loaded. */
int32_t sentinel_shadow_warm(void);

#ifdef __cplusplus
}
#endif

#endif
