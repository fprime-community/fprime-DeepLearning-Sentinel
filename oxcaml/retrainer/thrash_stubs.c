/* E2 apparatus. docs/MODELS.md 47.9.1. Same hand-written shape as retrainer_stubs.c. */
#include "sentinel_thrash.h"

#include <caml/mlvalues.h>
#include <caml/callback.h>
#include <caml/memory.h>
#include <caml/alloc.h>

static int32_t call0(const char *name)
{
    CAMLparam0();
    CAMLlocal1(result);
    const value *f = caml_named_value(name);
    if (f == NULL) {
        CAMLreturnT(int32_t, -1);
    }
    result = caml_callback(*f, Val_unit);
    CAMLreturnT(int32_t, (int32_t)Long_val(result));
}

int32_t sentinel_thrash_start_domain(void) { return call0("sentinel_thrash_start_domain"); }
int32_t sentinel_thrash_stop(void)         { return call0("sentinel_thrash_stop"); }
int32_t sentinel_thrash_count(void)        { return call0("sentinel_thrash_count"); }

int32_t sentinel_thrash_run_for(int32_t n, int32_t ms, const char *path)
{
    CAMLparam0();
    CAMLlocal3(tup, s, result);
    const value *f = caml_named_value("sentinel_thrash_run_for");
    if (f == NULL) {
        CAMLreturnT(int32_t, -1);
    }
    s = caml_copy_string(path);
    tup = caml_alloc_tuple(3);
    Store_field(tup, 0, Val_long((long)n));
    Store_field(tup, 1, Val_long((long)ms));
    Store_field(tup, 2, s);
    result = caml_callback(*f, tup);
    CAMLreturnT(int32_t, (int32_t)Long_val(result));
}
