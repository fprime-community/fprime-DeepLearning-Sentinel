/* D85: hand-written stubs over window_c.ml, in shadow_stubs.c's shape.
 *
 * Each boots first (idempotent; cycle_stubs.c owns the one runtime), then takes GC
 * roots, calls the registered closure and releases the roots. No OCaml value
 * outlives its CAMLreturn. No buffer crosses: every argument is an integer.
 */
#include "sentinel_window.h"
#include "sentinel_cycle.h"

#include <caml/mlvalues.h>
#include <caml/callback.h>
#include <caml/memory.h>

static int window_ready(void)
{
    return sentinel_cycle_boot() == SENTINEL_CYC_OK;
}

static int32_t call0(const char *name)
{
    if (!window_ready()) {
        return SENTINEL_WIN_ERR_NO_RUNTIME;
    }
    CAMLparam0();
    CAMLlocal1(r);
    const value *f = caml_named_value(name);
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_WIN_ERR_NO_RUNTIME);
    }
    r = caml_callback(*f, Val_unit);
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

int32_t sentinel_window_configure(int32_t nominal_ppm)
{
    if (!window_ready()) {
        return SENTINEL_WIN_ERR_NO_RUNTIME;
    }
    CAMLparam0();
    CAMLlocal1(r);
    const value *f = caml_named_value("sentinel_win_configure");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_WIN_ERR_NO_RUNTIME);
    }
    r = caml_callback(*f, Val_int(nominal_ppm));
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

int32_t sentinel_window_push(int32_t limit_any, int32_t emitted)
{
    if (!window_ready()) {
        return SENTINEL_WIN_ERR_NO_RUNTIME;
    }
    CAMLparam0();
    CAMLlocal1(r);
    const value *f = caml_named_value("sentinel_win_push");
    if (f == NULL) {
        CAMLreturnT(int32_t, SENTINEL_WIN_ERR_NO_RUNTIME);
    }
    r = caml_callback2(*f, Val_int(limit_any), Val_int(emitted));
    CAMLreturnT(int32_t, (int32_t)Int_val(r));
}

int32_t sentinel_window_admits(void) { return call0("sentinel_win_admits"); }
int32_t sentinel_window_emits(void) { return call0("sentinel_win_emits"); }
int32_t sentinel_window_limits(void) { return call0("sentinel_win_limits"); }
int32_t sentinel_window_reset(void) { return call0("sentinel_win_reset"); }
