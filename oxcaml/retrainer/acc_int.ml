(* THE FLOWN ACCESSOR, specialised to `int array`. Bounds-checked. D82.
 *
 * (!) WHY A SECOND FLOWN ACCESSOR EXISTS. `acc.ml` wraps `Array` polymorphically at
 * `'a array`, which defeats the representation specialisation OCaml does for arrays:
 * the generic path may allocate, and `[@zero_alloc strict]` then refuses the CALLER.
 * Measured 2026-09-22: nine of the eleven annotated modules hold `strict` against the
 * polymorphic wrapper and two do not -- `window56.ml` (two ring counters) and
 * `shadow59.ml` (a CRC table), whose arrays are `int array`.
 *
 * Against this file, specialised to `int`, both hold `strict` while fully
 * bounds-checked. **So the obstacle was the wrapper's polymorphism, not the bounds
 * checks**, and the conclusion is general: every annotated module in this tree holds
 * `[@zero_alloc strict]` with every access checked.
 *
 * It is not interchangeable with `acc.ml`: a module holding `float array` as well
 * needs that one. The split is by element type and is recorded here rather than left
 * for a reader to rediscover from a compiler error. *)

module Array = struct
  include Stdlib.Array

  let[@inline always] unsafe_get (a : int array) (i : int) : int = Stdlib.Array.get a i
  let[@inline always] unsafe_set (a : int array) (i : int) (v : int) : unit =
    Stdlib.Array.set a i v
end

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
