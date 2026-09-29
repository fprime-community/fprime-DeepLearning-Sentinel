(* THE FLOWN ACCESSOR. Bounds-checked, and it is what the retrainer now compiles
 * against. D82.
 *
 * (!) WHAT CHANGED AND WHY. Until 2026-09-22 this tree performed 559 unchecked array
 * accesses across 19 modules, including every module carrying `[@zero_alloc strict]`,
 * and nothing in the record said so. The OxCaml case was argued on the annotation;
 * a reader could reasonably have concluded the training arithmetic was bounds-checked.
 * It was not.
 *
 * The reasonable defence would have been that the annotation forced it: `strict`
 * refuses a function whose paths reach an exceptional return -- 47.9 recorded the
 * compiler's own "may allocate ON A PATH TO EXCEPTIONAL RETURN" -- and a checked
 * access raises `Invalid_argument`. **Measured, that defence does not hold.** OCaml's
 * bounds-failure path raises a PREALLOCATED exception and allocates nothing, so
 * `strict` is satisfied with the checks in place. The accesses were a choice, and D82
 * takes the other one.
 *
 * So: every element access in the retrainer is now bounds-checked, and every
 * `[@zero_alloc strict]` site still holds with 0 `assume`.
 *
 * (!) THE POLYMORPHISM LIMIT, WHICH IS REAL AND IS WHY `acc_int.ml` EXISTS. Wrapping
 * `Array` at `'a array` defeats the representation specialisation OCaml does for
 * arrays, and the generic path may allocate; `strict` then refuses the CALLER. It
 * shows up only where the element type differs from the majority, which is exactly
 * how a wrapper hides a defect. `window56.ml` and `shadow59.ml` hold `int array` and
 * take `acc_int.ml`; everything else takes this file.
 *
 * `acc_unchecked.ml` is retained as the comparison arm, not as a build option:
 * `scripts/oxcaml_checked.sh` compiles both and reports the difference, which is what
 * keeps "the checks are in and `strict` still holds" a measurement rather than a
 * claim.
 *
 * `[@inline always]` is load-bearing on every wrapper: an un-inlined call from an
 * annotated function is an indirect call, which 47.9 measured `strict` rejecting by
 * name ("called function may allocate (indirect call)").
 *
 * No timing figure is produced for what the checks cost. Stop 35. *)

module Array = struct
  include Stdlib.Array

  (* `get` / `set`, not `unsafe_get` / `unsafe_set`. That is the whole change, and the
     call sites keep their spelling so the 559 of them stay greppable. *)
  let[@inline always] unsafe_get (a : 'a array) (i : int) : 'a = Stdlib.Array.get a i

  let[@inline always] unsafe_set (a : 'a array) (i : int) (v : 'a) : unit =
    Stdlib.Array.set a i v
end

(* Monomorphic per kind: a polymorphic Bigarray wrapper compiles to the generic
   `caml_ba_get_1` external, and `strict` refuses the caller with "called function may
   allocate (external call to caml_ba_get_1)". *)

module F32 = struct
  type t = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

  let[@inline always] get (a : t) (i : int) : float = Bigarray.Array1.get a i
  let[@inline always] set (a : t) (i : int) (v : float) : unit =
    Bigarray.Array1.set a i v
end

module F64 = struct
  type t = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

  let[@inline always] get (a : t) (i : int) : float = Bigarray.Array1.get a i
  let[@inline always] set (a : t) (i : int) (v : float) : unit =
    Bigarray.Array1.set a i v
end

module U8 = struct
  type t = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

  let[@inline always] get (a : t) (i : int) : int = Bigarray.Array1.get a i
  let[@inline always] set (a : t) (i : int) (v : int) : unit =
    Bigarray.Array1.set a i v
end
