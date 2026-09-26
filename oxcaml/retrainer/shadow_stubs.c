/* Section 72 / E5-e: hand-written stubs, in cycle_stubs.c's shape.
 *
 * Every function takes the GC roots, looks up the closure, calls it, converts the
 * result to a fixed-size integer and releases the roots. No OCaml value outlives
 * its CAMLreturn, and none is stored anywhere in C or C++.
 *
 * (!) CAML_BA_EXTERNAL IS THE LOAD-BEARING FLAG. It tells the runtime the memory
 * belongs to the caller, so the GC neither frees it nor moves it, and the wrapper
 * is a view rather than a copy.
 *
 * (!) NO caml_startup HERE. cycle_stubs.c:24-38 owns the one runtime this object
 * carries; sentinel_cycle_boot() is idempotent, so calling it on the way in makes
 * each entry safe without starting a second runtime, which -output-complete-obj
 * would not permit anyway (scripts/oxcaml_s61.sh:79-85).
 */
#include "sentinel_shadow.h"
#include "sentinel_cycle.h"

#include <caml/mlvalues.h>
#include <caml/callback.h>
#include <caml/memory.h>
#include <caml/bigarray.h>

/* The kind, layout and ownership flags, widened once. See the note at the call
   sites: the flags come from two different enums and clang's
   -Wenum-enum-conversion rejects OR-ing them at -Werror. */
#define SENTINEL_BA_U8_EXTERNAL \
    ((int)CAML_BA_UINT8 | (int)CAML_BA_C_LAYOUT | (int)CAML_BA_EXTERNAL)
#define SENTINEL_BA_F32_EXTERNAL \
    ((int)CAML_BA_FLOAT32 | (int)CAML_BA_C_LAYOUT | (int)CAML_BA_EXTERNAL)

/* (!) THE BOOT MUST HAPPEN BEFORE CAMLparam0, NOT INSIDE IT. CAMLparam0 and
   CAMLlocal register GC roots, which reads the domain's state and therefore
   requires the domain lock to be held already. Booting from inside the macro
   block aborts the process with "Fatal error: no domain lock held" -- measured
   at Section 72, and it is the same family as 65.10's finding, one level down:
   65.10 was about WHICH THREAD holds the lock, this is about WHEN it is taken.
   So every entry point below boots first, on its own caller's thread, and only
   then takes roots. cycle_stubs.c never hit this because its boot is called out
   of band by Retrainer.cpp before any entry point runs. */
static int shadow_ready(void)
{
    return sentinel_cycle_boot() == SENTINEL_CYC_OK;
}

int32_t sentinel_shadow_load(const uint8_t *flying, uint32_t n)
{
    if (!shadow_ready()) {
        return SENTINEL_SHD_ERR_NO_RUNTIME;
    }
    CAMLparam0();
    CAMLlocal2(ba, r);
    const value *f = caml_named_value("sentinel_shd_load");
    if ((f == NULL) || (flying == NULL)) {
        CAMLreturnT(int32_t, SENTINEL_SHD_ERR_NO_RUNTIME);
    }
    intnat dims[1];
    dims[0] = (intnat)n;
    /* The caller owns it. The view dies with this call. */
    ba = caml_ba_alloc(SENTINEL_BA_U8_EXTERNAL,
                       1, (void *)(uintptr_t)flying, dims);
    r = caml_callback(*f, ba);
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

int32_t sentinel_shadow_write(const float *weights, uint32_t n)
{
    if (!shadow_ready()) {
        return SENTINEL_SHD_ERR_NO_RUNTIME;
    }
    CAMLparam0();
    CAMLlocal2(ba, r);
    const value *f = caml_named_value("sentinel_shd_write");
    if ((f == NULL) || (weights == NULL)) {
        CAMLreturnT(int32_t, SENTINEL_SHD_ERR_NO_RUNTIME);
    }
    intnat dims[1];
    dims[0] = (intnat)n;
    ba = caml_ba_alloc(SENTINEL_BA_F32_EXTERNAL,
                       1, (void *)(uintptr_t)weights, dims);
    r = caml_callback(*f, ba);
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

int32_t sentinel_shadow_export(uint8_t *out, uint32_t cap)
{
    if (!shadow_ready()) {
        return SENTINEL_SHD_ERR_NO_RUNTIME;
    }
    CAMLparam0();
    CAMLlocal2(ba, r);
    const value *f = caml_named_value("sentinel_shd_export");
    if ((f == NULL) || (out == NULL)) {
        CAMLreturnT(int32_t, SENTINEL_SHD_ERR_NO_RUNTIME);
    }
    intnat dims[1];
    dims[0] = (intnat)cap;
    ba = caml_ba_alloc(SENTINEL_BA_U8_EXTERNAL, 1, (void *)out, dims);
    r = caml_callback(*f, ba);
    /* Positive is the byte count written; negative is a refusal. */
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

int32_t sentinel_shadow_loss(float *out, uint32_t n)
{
    if (!shadow_ready()) {
        return SENTINEL_SHD_ERR_NO_RUNTIME;
    }
    CAMLparam0();
    CAMLlocal2(ba, r);
    const value *f = caml_named_value("sentinel_shd_loss");
    if ((f == NULL) || (out == NULL)) {
        CAMLreturnT(int32_t, SENTINEL_SHD_ERR_NO_RUNTIME);
    }
    intnat dims[1];
    dims[0] = (intnat)n;
    ba = caml_ba_alloc(SENTINEL_BA_F32_EXTERNAL, 1, (void *)out, dims);
    r = caml_callback(*f, ba);
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

/* D85: the onboard warm start. No buffer crosses; the weights are already inside
   the OCaml side, loaded by sentinel_shadow_load. */
int32_t sentinel_shadow_warm(void)
{
    if (!shadow_ready()) {
        return SENTINEL_SHD_ERR_NO_RUNTIME;
    }
    CAMLparam0();
    CAMLlocal1(r);
    const value *f = caml_named_value("sentinel_shd_warm");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_SHD_ERR_NO_RUNTIME);
    }
    r = caml_callback(*f, Val_unit);
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

