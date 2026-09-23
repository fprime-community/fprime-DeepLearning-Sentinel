(* Section 63, OW4: E3's `checksum`, written with OxCaml's unboxed `float#`.
 *
 * 47.13.4 recorded ONE accommodation in `zalloc.ml`: written the obvious way, returning
 * its float, the checker refused `checksum` with "Error: allocation of 16 bytes for
 * float" -- a float RETURNED across a non-inlined boundary is boxed. It was changed to
 * write into a preallocated slot. `zalloc.ml:83-85` records that OxCaml's unboxed
 * `float#` is the language-native alternative and was DELIBERATELY not used, because E3
 * asked what plain OCaml costs under `strict`.
 *
 * This module asks the other question, which 47.14 registered as owed: with `float#`,
 * does the accommodation disappear, so E3 reads four of four rather than three and an
 * accommodation?
 *
 * (!) THE OPERATIONS ARRIVE THROUGH AN INSTALLED, VERSIONED LIBRARY, which is exactly
 * what 54.9 said would settle the undocumented-primitive dependency:
 * `Stdlib_upstream_compatible.Float_u`, version 5.2.0+ox, `float_u.mli:49` `type t =
 * float#`, `:68-77` add/sub/mul/div, `:56` `to_float = "%box_float"`, `:59` `of_float =
 * "%unbox_float"`. Not an undeclared compiler primitive.
 *
 * (!) AND ONE LIMITATION IS RECORDED RATHER THAN WORKED AROUND SILENTLY. `float#` cannot
 * live in a `ref` at this version -- `let s = ref #0.0` is refused with "The layout of
 * float# is float64" against an expected "value_or_null". `float_u.mli:51-52` says why in
 * its own words: "CR layouts v5: add back all the constants in this module (e.g., [zero]
 * and [infinity]) when we we support [float64]s in structures." So the accumulator is a
 * tail-recursive parameter. That is the language-native shape for an unboxed
 * accumulator, not a contortion -- but it IS a second structural constraint, and a
 * reader weighing `float#` should meet it here rather than discover it. *)


(* D82: every element access below is bounds-checked. `acc.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and all 56 `[@zero_alloc strict]` sites still
   hold. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc

module F = Stdlib_upstream_compatible.Float_u

let n = 16
let c = Array.make (n * n) 0.0
let w = Array.make (n * n) 0.0
let out = Array.make 1 0.0

(* The slot form, `zalloc.ml:88-93` verbatim. The control. *)
let[@zero_alloc strict] checksum_slot () =
  let s = ref 0.0 in
  for i = 0 to (n * n) - 1 do
    s := !s +. (Array.unsafe_get c i) +. (Array.unsafe_get w i)
  done;
  Array.unsafe_set out 0 !s

(* The unboxed form. It RETURNS its float, which is the thing 47.13.4 could not do.
 *
 * (!) THE ASSOCIATION IS THE SLOT FORM'S, DELIBERATELY. `s := !s +. c.(i) +. w.(i)`
 * parses as `((s + c.(i)) + w.(i))` because `+.` is left-associative, so this writes
 * `F.add (F.add acc c_i) w_i` and NOT `F.add acc (F.add c_i w_i)`. Those differ in the
 * last bit, and a mismatched association here would report as a value difference that
 * had nothing to do with boxing -- the instrument failing for a reason unrelated to what
 * it tests, which is the mistake 47.13.4's own first probe made. *)
let[@zero_alloc strict] checksum_unboxed () : float# =
  let rec go i (acc : float#) : float# =
    if i >= n * n then acc
    else
      go (i + 1)
        (F.add (F.add acc (F.of_float (Array.unsafe_get c i)))
           (F.of_float (Array.unsafe_get w i)))
  in
  go 0 #0.0
