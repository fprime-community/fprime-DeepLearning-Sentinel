(* THE COUNTING ACCESSOR. D82.1's cost proxy: how many bounds checks a retraining
 * cycle actually executes, counted rather than timed.
 *
 * (!) WHY COUNTED AND NOT TIMED. D82 c.5 left the run-time cost of bounds-checking
 * unquantified because stop 35 forbids quoting a timing figure from this work, and
 * D82.1 keeps that: the wall-clock belongs to the flight-hardware session with the
 * rest of the timing work, not to a laptop. A COUNT is not a timing figure. It is
 * exact, deterministic, reproducible on any host, and it is the quantity a reader
 * actually wants when asking what the checks cost -- the per-tick work, in the same
 * currency as `docs/MODELS.md` 19.8 F4's **70,080 MAC per tick** for the detector.
 *
 * Same interface as `acc.ml`, same bounds checks, plus one counter per access. This
 * is a TEST-BUILD accessor: `acc.ml` is what flies and carries no counter.
 *
 * The counter is an `int ref` incremented in place. It allocates nothing, so a module
 * built against this file still satisfies `[@zero_alloc strict]` -- which
 * `scripts/oxcaml_count.sh` checks rather than assumes, because a cost proxy that
 * perturbed the property it is measuring would be worthless. *)

let checks = ref 0
let reset () = checks := 0
let count () = !checks

module Array = struct
  include Stdlib.Array

  let[@inline always] unsafe_get (a : 'a array) (i : int) : 'a =
    incr checks; Stdlib.Array.get a i

  let[@inline always] unsafe_set (a : 'a array) (i : int) (v : 'a) : unit =
    incr checks; Stdlib.Array.set a i v
end

module F32 = struct
  type t = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

  let[@inline always] get (a : t) (i : int) : float =
    incr checks; Bigarray.Array1.get a i

  let[@inline always] set (a : t) (i : int) (v : float) : unit =
    incr checks; Bigarray.Array1.set a i v
end

module F64 = struct
  type t = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

  let[@inline always] get (a : t) (i : int) : float =
    incr checks; Bigarray.Array1.get a i

  let[@inline always] set (a : t) (i : int) (v : float) : unit =
    incr checks; Bigarray.Array1.set a i v
end

module U8 = struct
  type t = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

  let[@inline always] get (a : t) (i : int) : int =
    incr checks; Bigarray.Array1.get a i

  let[@inline always] set (a : t) (i : int) (v : int) : unit =
    incr checks; Bigarray.Array1.set a i v
end
