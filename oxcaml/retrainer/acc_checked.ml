(* The CHECKED accessor. Byte-for-byte the same interface as `acc.ml`, bounds-checked
 * underneath. A build selects it by copying it in as `acc.ml`;
 * `scripts/oxcaml_checked.sh` is the runner that does so for the whole ladder.
 *
 * (!) THIS IS NOT WHAT FLIES, AND THE REASON IS NOT THE ONE EXPECTED. The reasonable
 * guess was that `[@zero_alloc strict]` could not survive bounds-checking, because
 * `strict` refuses a function whose paths reach an exceptional return -- 47.9 measured
 * the compiler saying "may allocate ON A PATH TO EXCEPTIONAL RETURN" -- and a checked
 * access raises `Invalid_argument`. **That guess is refuted.** Measured 2026-09-22 on
 * switch 5.2.0+ox: `deep_f32.ml`, the whole 75,360-parameter cycle, compiles clean
 * under `-zero-alloc-check all` against THIS file, with every array and Bigarray
 * access checked. OCaml's bounds-failure path raises a preallocated exception and does
 * not allocate, so `strict` is satisfied either way.
 *
 * So the flown object being unchecked is a CHOICE, not a constraint the language
 * imposed, and this file exists so the choice can be re-costed rather than assumed.
 * What is not claimed here: any figure for what the checking costs at run time. Stop
 * 35 forbids quoting a timing figure from this work, and none is produced.
 *
 * The one thing this variant buys that `acc.ml` cannot: an index error in the training
 * arithmetic stops the run with `Invalid_argument` instead of reading or writing past
 * an array. `scripts/oxcaml_checked.sh` proves that direction by injecting one. *)

module Array = struct
  include Stdlib.Array

  (* `get` / `set`, not `unsafe_get` / `unsafe_set`. That is the whole difference. *)
  let[@inline always] unsafe_get (a : 'a array) (i : int) : 'a = Stdlib.Array.get a i

  let[@inline always] unsafe_set (a : 'a array) (i : int) (v : 'a) : unit =
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
