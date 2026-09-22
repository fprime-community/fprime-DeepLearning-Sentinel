(* The CHECKED accessor, specialised to `int array`.
 *
 * (!) WHY A SECOND CHECKED FILE EXISTS, AND IT IS THE INTERESTING HALF OF THE RESULT.
 * `acc_checked.ml` wraps `Array` polymorphically at `'a array`. Measured 2026-09-22:
 * nine of the eleven annotated modules hold `[@zero_alloc strict]` against it, and two
 * do not -- `window56.ml` (two ring counters) and `shadow59.ml` (a CRC table), whose
 * arrays are `int array`. Wrapping at `'a array` defeats the representation
 * specialisation OCaml does for arrays, and the generic path may allocate, so `strict`
 * refuses the caller.
 *
 * Against THIS file, specialised to `int`, **both of those modules hold `strict` while
 * fully bounds-checked.** So the obstacle is the wrapper's polymorphism, not
 * bounds-checking, and the conclusion stands for the whole ladder: `[@zero_alloc
 * strict]` and bounds-checked access coexist. Reaching it needs an accessor that
 * specialises to the element type, which is mechanical rather than a limit of the
 * language.
 *
 * It is not a drop-in for the other nine: they hold `float array` as well, and a
 * module holding both needs more than one specialisation. That is recorded rather
 * than solved here -- solving it is a change to the modules, and 72.4 forbids editing
 * a measured module without re-earning its figures. *)

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
