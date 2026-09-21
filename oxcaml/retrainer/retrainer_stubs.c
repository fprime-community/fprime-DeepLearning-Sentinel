/* E1, the hand-written stubs. docs/MODELS.md 47.6.
 *
 * (!) A NOTE ON "CAMLprim", BECAUSE THE BRIEF'S WORD FOR THIS NAMES THE OTHER DIRECTION.
 * `CAMLprim` marks a C function that OCaml calls through an `external` declaration.
 * E1 needs the opposite: C++ calling INTO OCaml. That direction goes through
 * `caml_named_value` on a closure the OCaml side registered with `Callback.register`,
 * and there is no `CAMLprim` in it. What the brief was asking for -- hand-written stubs
 * rather than a generated FFI, so every crossing is reviewable source -- is what this
 * file is. Recorded rather than silently substituted.
 *
 * Every function here follows the same shape: take the GC roots, look up the closure,
 * call it, convert the result to a fixed-size integer, release the roots. No OCaml value
 * outlives its CAMLreturn.
 */
#include "sentinel_retrainer.h"

#include <caml/mlvalues.h>
#include <caml/callback.h>
#include <caml/memory.h>
#include <caml/bigarray.h>
#include <caml/threads.h>

static int g_booted = 0;

int32_t sentinel_retrainer_boot(void)
{
    if (g_booted) {
        return SENTINEL_RT_OK;
    }
    /* argv[0] is conventional and unused; the array must be NULL-terminated. */
    static char  arg0[] = "sentinel_retrainer";
    static char *argv[] = { arg0, NULL };
    caml_startup(argv);
    g_booted = 1;
    /* If the OCaml module's toplevel did not run, the closures are absent and every
     * call below would be a null dereference. Checked once, here. */
    if (caml_named_value("sentinel_rt_init") == NULL) {
        g_booted = 0;
        return SENTINEL_RT_ERR_NO_RUNTIME;
    }
    return SENTINEL_RT_OK;
}

/* Looks up a registered closure, returning NULL if the runtime is not up. */
static const value *closure(const char *name)
{
    if (!g_booted) {
        return NULL;
    }
    return caml_named_value(name);
}

int32_t sentinel_retrainer_init(uint32_t capacity)
{
    CAMLparam0();
    CAMLlocal1(result);
    const value *f = closure("sentinel_rt_init");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_RT_ERR_NO_RUNTIME);
    }
    /* Int_val/Val_long carry 63 bits here, so a uint32_t always fits. */
    result = caml_callback(*f, Val_long((long)capacity));
    CAMLreturnT(int32_t, (int32_t)Long_val(result));
}

int32_t sentinel_retrainer_feed(const double *samples, uint32_t n)
{
    CAMLparam0();
    CAMLlocal2(ba, result);
    const value *f = closure("sentinel_rt_feed");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_RT_ERR_NO_RUNTIME);
    }
    if (samples == NULL) {
        CAMLreturnT(int32_t, SENTINEL_RT_ERR_SHORT_BUFFER);
    }
    /* (!) CAML_BA_EXTERNAL is the whole point. The buffer belongs to the caller, lives
     * outside the OCaml heap, is never moved by the GC and is never freed by OCaml --
     * which is why docs/MODELS.md 47.6 names Bigarray as the telemetry carrier at this
     * boundary and nothing else. The cast drops const because the Bigarray API takes a
     * void*; the OCaml side only reads it. */
    {
        intnat dim = (intnat)n;
        ba = caml_ba_alloc_dims(CAML_BA_FLOAT64 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL,
                                1, (void *)(uintptr_t)samples, dim);
        result = caml_callback(*f, ba);
    }
    CAMLreturnT(int32_t, (int32_t)Long_val(result));
}

int32_t sentinel_retrainer_step(void)
{
    CAMLparam0();
    CAMLlocal1(result);
    const value *f = closure("sentinel_rt_step");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_RT_ERR_NO_RUNTIME);
    }
    result = caml_callback(*f, Val_unit);
    CAMLreturnT(int32_t, (int32_t)Long_val(result));
}

int32_t sentinel_retrainer_status(void)
{
    CAMLparam0();
    CAMLlocal1(result);
    const value *f = closure("sentinel_rt_status");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_RT_ERR_NO_RUNTIME);
    }
    result = caml_callback(*f, Val_unit);
    CAMLreturnT(int32_t, (int32_t)Long_val(result));
}

int32_t sentinel_retrainer_export(double *out, uint32_t n)
{
    CAMLparam0();
    CAMLlocal2(ba, result);
    const value *f = closure("sentinel_rt_export");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_RT_ERR_NO_RUNTIME);
    }
    if (out == NULL) {
        CAMLreturnT(int32_t, SENTINEL_RT_ERR_SHORT_BUFFER);
    }
    {
        intnat dim = (intnat)n;
        ba = caml_ba_alloc_dims(CAML_BA_FLOAT64 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL,
                                1, (void *)out, dim);
        result = caml_callback(*f, ba);
    }
    CAMLreturnT(int32_t, (int32_t)Long_val(result));
}
