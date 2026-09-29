(* THE COMPARISON ARM. Unchecked, and NOT what flies any more -- D82 made the
 * bounds-checked `acc.ml` the flown accessor.
 *
 * This file is retained for one purpose: `scripts/oxcaml_checked.sh` compiles the
 * annotated modules against both accessors and reports the difference. That is what
 * keeps "every access is checked and every `[@zero_alloc strict]` site still hold"
 * a measurement rather than a claim, and it is the arm that would show a regression
 * if a future change made the checks unaffordable.
 *
 * It also preserves the negative direction: a read past the end returns whatever is
 * there under this accessor, and raises `Invalid_argument` under `acc.ml`. A pair of
 * builds that behave identically would mean the checks were not doing anything. *)

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
