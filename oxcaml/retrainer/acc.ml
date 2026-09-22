(* The FLIGHT accessor. Unchecked, and deliberately compiling to exactly what every
 * module here emitted before this file existed.
 *
 * (!) WHY THIS FILE EXISTS. `oxcaml/retrainer/` performs 559 unchecked array accesses
 * -- 268 `Array.unsafe_get`, 221 `Array.unsafe_set`, and 31 / 39 of the
 * `Bigarray.Array1` forms -- across 19 modules, including every module carrying
 * `[@zero_alloc strict]`. Nothing in `docs/`, `README.md` or `master:docs/DESIGN.md`
 * said so. A reader of the OxCaml case could reasonably have concluded the training
 * arithmetic was bounds-checked. It was not, and it is not the compiler's doing: the
 * accesses were written unchecked and nothing recorded the choice or its cost.
 *
 * This file makes the choice selectable instead of implicit. `acc_checked.ml` is the
 * same interface with `Array.get` / `Array.set` underneath; a build picks one by
 * copying it in as `acc.ml`, and a module opts in with one line -- `open Acc` -- so
 * all 489 `Array.*` call sites stay exactly as they were, and stay greppable.
 *
 * (!) AND THE MEASURED RESULT IS THE OPPOSITE OF WHAT WAS EXPECTED. The reasonable
 * guess was that `strict` and bounds-checking could not coexist: `strict` refuses a
 * function whose paths reach an exceptional return (47.9 recorded the compiler's own
 * "may allocate ON A PATH TO EXCEPTIONAL RETURN"), and a checked access raises
 * `Invalid_argument`. Measured 2026-09-22 on switch 5.2.0+ox: **`deep_f32.ml` -- the
 * whole 75,360-parameter training cycle -- compiles clean under
 * `-zero-alloc-check all` with every array AND every Bigarray access bounds-checked.**
 * The bounds-failure path does not allocate, so `strict` is satisfied. The two
 * properties are not in tension, and the unchecked accesses were never required by
 * the annotation. `scripts/oxcaml_checked.sh` is that result, re-derived on demand.
 *
 * (!) WHAT A GENERIC WRAPPER CANNOT DO, AND IT IS A REAL LIMIT. Wrapping
 * `Bigarray.Array1` polymorphically defeats flambda2's kind specialisation: the
 * wrapper compiles to the generic `caml_ba_get_1` external, and `strict` then refuses
 * the caller with "called function may allocate (external call to caml_ba_get_1)".
 * So the Bigarray helpers below are MONOMORPHIC, one per kind in use. Call sites that
 * still spell `Bigarray.Array1.unsafe_get` directly are unaffected by this file and
 * remain unchecked in both variants; `docs/MODELS.md` records which and why.
 *
 * `[@inline always]` is load-bearing on every wrapper: an un-inlined call from an
 * annotated function is an indirect call, which 47.9 measured `strict` rejecting by
 * name ("called function may allocate (indirect call)"). *)

(* (!) NO WRAPPER AT ALL, AND THAT IS DELIBERATE. A polymorphic
 * `let[@inline always] unsafe_get (a : 'a array) i = Stdlib.Array.unsafe_get a i`
 * was tried first and REJECTED by `strict` in `window56.ml` and `shadow59.ml`, whose
 * arrays are `int array`: wrapping at `'a array` defeats the representation
 * specialisation OCaml does for flat float arrays, and the generic path may allocate.
 * The other nine modules held, which is exactly how a wrapper like this hides a
 * defect -- it fails only where the element type differs.
 *
 * So the flight accessor re-exports the module unchanged. `Array.unsafe_get` here IS
 * `Stdlib.Array.unsafe_get`, compiled to the same instruction as before this file
 * existed, and the flight object is provably unperturbed. Only `acc_checked.ml`
 * substitutes anything. *)
module Array = Stdlib.Array

(* Monomorphic per kind. See the note above: a polymorphic Bigarray wrapper does not
 * specialise and takes `[@zero_alloc strict]` down with it. *)

module F32 = struct
  type t = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

  let[@inline always] get (a : t) (i : int) : float = Bigarray.Array1.unsafe_get a i
  let[@inline always] set (a : t) (i : int) (v : float) : unit =
    Bigarray.Array1.unsafe_set a i v
end

module F64 = struct
  type t = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

  let[@inline always] get (a : t) (i : int) : float = Bigarray.Array1.unsafe_get a i
  let[@inline always] set (a : t) (i : int) (v : float) : unit =
    Bigarray.Array1.unsafe_set a i v
end

module U8 = struct
  type t = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

  let[@inline always] get (a : t) (i : int) : int = Bigarray.Array1.unsafe_get a i
  let[@inline always] set (a : t) (i : int) (v : int) : unit =
    Bigarray.Array1.unsafe_set a i v
end
