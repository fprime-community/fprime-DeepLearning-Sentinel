/* Section 61 / E5-c: hand-written stubs, in retrainer_stubs.c's shape.
 *
 * Every function takes the GC roots, looks up the closure, calls it, converts the
 * result to a fixed-size integer and releases the roots. No OCaml value outlives
 * its CAMLreturn, and none is stored anywhere in C or C++.
 *
 * (!) CAML_BA_EXTERNAL IS THE LOAD-BEARING FLAG. It tells the runtime the memory
 * belongs to the caller, so the GC neither frees it nor moves it, and the wrapper
 * is a view rather than a copy.
 */
#include "sentinel_cycle.h"

#include <caml/mlvalues.h>
#include <caml/callback.h>
#include <caml/memory.h>
#include <caml/bigarray.h>

/* The kind, layout and ownership flags, widened once. See the note at the call sites. */
#define SENTINEL_BA_F32_EXTERNAL \
    ((int)CAML_BA_FLOAT32 | (int)CAML_BA_C_LAYOUT | (int)CAML_BA_EXTERNAL)

static int g_booted = 0;

int32_t sentinel_cycle_boot(void)
{
    if (g_booted) {
        return SENTINEL_CYC_OK;
    }
    static char  arg0[] = "sentinel_cycle";
    static char *argv[] = { arg0, NULL };
    caml_startup(argv);
    g_booted = 1;
    if (caml_named_value("sentinel_cyc_init") == NULL) {
        g_booted = 0;
        return SENTINEL_CYC_ERR_NO_RUNTIME;
    }
    return SENTINEL_CYC_OK;
}

static const value *closure(const char *name)
{
    if (!g_booted) {
        return NULL;
    }
    return caml_named_value(name);
}

int32_t sentinel_cycle_init(int32_t salt)
{
    CAMLparam0();
    CAMLlocal1(r);
    const value *f = closure("sentinel_cyc_init");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_CYC_ERR_NO_RUNTIME);
    }
    r = caml_callback(*f, Val_int((int)salt));
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

int32_t sentinel_cycle_load(const float *window, uint32_t n)
{
    CAMLparam0();
    CAMLlocal2(ba, r);
    const value *f = closure("sentinel_cyc_load");
    if ((f == NULL) || (window == NULL)) {
        CAMLreturnT(int32_t, SENTINEL_CYC_ERR_NO_RUNTIME);
    }
    intnat dims[1];
    dims[0] = (intnat)n;
    /* The caller owns it. The view dies with this call. */
    /* (!) THE FLAGS COME FROM TWO DIFFERENT ENUMS -- caml_ba_kind and
       caml_ba_layout -- and OR-ing them is what the runtime's own API asks for.
       Clang's -Wenum-enum-conversion rejects that at -Werror, so each is widened
       to int explicitly. The framework's flag set is not relaxed to accommodate
       the runtime; the crossing is written to satisfy it. */
    ba = caml_ba_alloc(SENTINEL_BA_F32_EXTERNAL,
                       1, (void *)(uintptr_t)window, dims);
    r = caml_callback(*f, ba);
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

int32_t sentinel_cycle_run(int32_t budget, int32_t t_steps, int32_t admit)
{
    CAMLparam0();
    CAMLlocal1(r);
    const value *f = closure("sentinel_cyc_run");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_CYC_ERR_NO_RUNTIME);
    }
    r = caml_callback3(*f, Val_int((int)budget), Val_int((int)t_steps),
                       Val_int((int)admit));
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

int32_t sentinel_cycle_steps(void)
{
    CAMLparam0();
    CAMLlocal1(r);
    const value *f = closure("sentinel_cyc_steps");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_CYC_ERR_NO_RUNTIME);
    }
    r = caml_callback(*f, Val_unit);
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

int32_t sentinel_cycle_export(float *out, uint32_t n)
{
    CAMLparam0();
    CAMLlocal2(ba, r);
    const value *f = closure("sentinel_cyc_export");
    if ((f == NULL) || (out == NULL)) {
        CAMLreturnT(int32_t, SENTINEL_CYC_ERR_NO_RUNTIME);
    }
    intnat dims[1];
    dims[0] = (intnat)n;
    /* (!) THE FLAGS COME FROM TWO DIFFERENT ENUMS -- caml_ba_kind and
       caml_ba_layout -- and OR-ing them is what the runtime's own API asks for.
       Clang's -Wenum-enum-conversion rejects that at -Werror, so each is widened
       to int explicitly. The framework's flag set is not relaxed to accommodate
       the runtime; the crossing is written to satisfy it. */
    ba = caml_ba_alloc(SENTINEL_BA_F32_EXTERNAL,
                       1, (void *)out, dims);
    r = caml_callback(*f, ba);
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}
