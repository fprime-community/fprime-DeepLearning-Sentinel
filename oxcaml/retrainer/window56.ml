(* Section 56: what counts as recent healthy telemetry, decided onboard.
 *
 * (!) THIS MODULE SEES TWO BITS PER TICK AND NOTHING ELSE: whether any channel was
 * outside any limit band, and whether the flying detector emitted. It cannot see a
 * label, a fault time, a seed, or whether the run it is reading is the seeded one
 * or its control. Stop 25 requires that, and the type signature is the enforcement:
 * `push : int -> int -> unit`. Ground truth belongs to the SCORER.
 *
 * Three gates over a trailing window of W ticks, all causal (56.3):
 *   G-limit  no tick in the window was outside any band, of any colour. The bands
 *            are the spacecraft's engineering dictionary, not the detector's
 *            opinion, which is what breaks candidate (a)'s circularity as far as
 *            it can be broken.
 *   G-rate   emissions in the window <= k x the model's own calibrated nominal
 *            rate x W. 42.9's held-out 0.1830%, k = 2. Neither is fitted here.
 *   G-suff   the window is full. Objective.md 10.2 fix 1: refuse, do not overstate.
 *
 * State is two ring counters, so the rule is O(1) per tick in fixed memory and
 * reads no tick later than the one it is deciding. Objective.md:986 rule 5. *)


(* D82: every element access below is bounds-checked. `acc_int.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and every `[@zero_alloc strict]` site still
   holds. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc_int

let w = 6550                    (* 42.3 departure 2's data floor *)
let nominal = 0.001830          (* 42.9's held-out sanity rate *)
let k = 2.0                     (* 56.3: beyond the 1.75x the ground already saw *)

(* 24. Computed once at init, not per tick. *)
let ceiling = int_of_float (ceil (k *. nominal *. float_of_int w))

(* D85: THE NOMINAL RATE IS THE FLYING MODEL'S OWN, set once at init. 56's G-rate is
   defined as "k x the model's own calibrated nominal rate x W"; 0.001830 is 42.9's
   model's figure, and a mission flies its own. `set_nominal_ppm` recomputes the
   ceiling from the model's pre-launch figure BEFORE the first push. W and k do not
   move (stop 24), and unset, the ceiling is 42.9's 24 exactly as 56 measured it. *)
let ceil_slot = Array.make 1 ceiling

let set_nominal_ppm ppm =
  Array.unsafe_set ceil_slot 0
    (int_of_float (ceil (k *. (float_of_int ppm /. 1.0e6) *. float_of_int w)))

let limit_ring = Array.make w 0
let emit_ring = Array.make w 0

(* Slots, not returns: 48's G1 showed a float return boxes, and an int tuple would
   allocate. head, filled, limits in window, emissions in window. *)
let st = Array.make 4 0

let[@zero_alloc strict] reset () =
  for i = 0 to w - 1 do
    Array.unsafe_set limit_ring i 0; Array.unsafe_set emit_ring i 0
  done;
  for i = 0 to 3 do Array.unsafe_set st i 0 done

(* One tick. `lim` and `emit` are 0 or 1 and nothing else is passed in. *)
let[@zero_alloc strict] push lim emit =
  let head = Array.unsafe_get st 0 in
  (* evict what this slot held before admitting the new sample *)
  Array.unsafe_set st 2 ((Array.unsafe_get st 2) - (Array.unsafe_get limit_ring head));
  Array.unsafe_set st 3 ((Array.unsafe_get st 3) - (Array.unsafe_get emit_ring head));
  Array.unsafe_set limit_ring head lim;
  Array.unsafe_set emit_ring head emit;
  Array.unsafe_set st 2 ((Array.unsafe_get st 2) + lim);
  Array.unsafe_set st 3 ((Array.unsafe_get st 3) + emit);
  Array.unsafe_set st 0 ((head + 1) mod w);
  if (Array.unsafe_get st 1) < w then
    Array.unsafe_set st 1 ((Array.unsafe_get st 1) + 1)

let[@inline] full () = (Array.unsafe_get st 1) >= w
let[@inline] limits () = Array.unsafe_get st 2
let[@inline] emits () = Array.unsafe_get st 3

(* The rule. 1 admits, 0 refuses. *)
let[@zero_alloc strict] admits_all () =
  if full () && (limits () = 0) && (emits () <= Array.unsafe_get ceil_slot 0) then 1 else 0

(* WS4's comparison arm and WS7's control: the same rule with G-limit removed. *)
let[@zero_alloc strict] admits_rate_only () =
  if full () && (emits () <= Array.unsafe_get ceil_slot 0) then 1 else 0

(* Candidate (a) alone, read literally: "windows the flying detector was quiet
   through" is a window with NO emission in it. *)
let[@zero_alloc strict] admits_quiet_only () =
  if full () && (emits () = 0) then 1 else 0

(* G-suff as a refusal in its own right: a candidate shorter than the floor admits
   nothing, whatever the other two gates say. WS6. *)
let[@zero_alloc strict] sufficient () = if full () then 1 else 0
