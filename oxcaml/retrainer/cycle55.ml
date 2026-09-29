(* Section 55 arm B: a training cycle under D73's FIXED STEP BUDGET, in float32.
 *
 * (!) WHAT THIS SECTION TESTS IS DETERMINISM AND THE BUDGET, NOT THE OPTIMISER.
 * The parameter update here is a plain SGD step, and that is a deliberate scope
 * decision rather than a simplification:
 *
 *   An F32 Adam needs a float32 SQUARE ROOT. 54.2 measured %sqrtfloat32 as
 *   REJECTED -- "Unknown builtin primitive" -- so it would have to arrive as a C
 *   external `sqrtf`, and reaching for a float32 operation not in 54.2a's list is
 *   exactly stop 21's shape. 53's Adam is float64 and 54.6 forbids quoting it
 *   beside an F32 figure. So the F32 optimiser is E5-b's rung, pre-registered in
 *   its own right, and this loop uses only 54.2a's permitted set:
 *
 *     %addfloat32  %mulfloat32  %floatoffloat32  %float32offloat
 *
 *   %divfloat32 is permitted and is not needed here. No C external is called.
 *
 * (!) THE RNG IS ON THE FLIGHT-ADJACENT PATH AND 55.4 DISCLOSES IT. It is
 * counter-based: the mask for step k, unit i is a pure function of (seed, k, i),
 * so two processes draw identically and a resumed cycle draws identically. It is
 * never seeded from a clock or any entropy source. Integer arithmetic only, so it
 * allocates nothing. Objective.md:986 rule 5 is satisfied literally rather than
 * by argument. *)


(* D82: every element access below is bounds-checked. `acc.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and every `[@zero_alloc strict]` site still
   holds. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc

external to_f : float32 -> float = "%floatoffloat32"
external of_f : float -> float32 = "%float32offloat"
external add : float32 -> float32 -> float32 = "%addfloat32"
external mul : float32 -> float32 -> float32 = "%mulfloat32"

let hs = 80
let ins = 16

(* 55.4: EXPERIMENTAL, and stop 23 forbids reading a flight budget out of it. *)
let budget = 256

let lr = -0.01                            (* applied as p + (-lr)*g, one mul *)

(* lstm.py:92 dropout 0.3, as an exact integer ratio so no float rounding decides
   a mask. A unit is KEPT when rng*10 >= 3*2^30. *)
let drop_num = 3
let drop_den = 10
let rng_span = 0x40000000                 (* 2^30 *)
let rng_mask = 0x3FFFFFFF

(* Counter-based. No state, so nothing to reset and nothing to get out of step. *)
let[@inline] rng seed k i =
  let x = (seed * 0x9E3779B1) lxor ((k + 1) * 0x85EBCA6B) lxor ((i + 1) * 0xC2B2AE35) in
  let x = x lxor (x lsr 15) in
  let x = (x * 0x27D4EB2F) land rng_mask in
  x

let[@inline] keep seed k i = (rng seed k i) * drop_den >= drop_num * rng_span

(* Slots, not returns: 48's G1 showed a float return boxes. *)
let steps_taken = Array.make 1 0
let masked_units = Array.make 1 0

(* A deterministic, bounded drive. Not telemetry; the point is repeatability.
   DeterminismTest.cpp:47-49 uses the same shape for the same reason. *)
let[@inline] drive t c =
  let m = ((t * 2654435761) + (c * 40503)) land 0xFFFF in
  of_f (float_of_int m /. 65535.0)

(* One training cycle. `n_samples` is the SIZE OF THE TRAINING SET and changes
   which samples are seen; it does NOT change how many steps are taken. That is
   D73, and DT4 is the check. *)
let[@zero_alloc strict] run_cycle n_samples seed =
  Array.unsafe_set steps_taken 0 0;
  Array.unsafe_set masked_units 0 0;
  for k = 0 to budget - 1 do
    let s = k mod n_samples in
    for j = 0 to ins - 1 do
      F32.set Gru_f32.x j (to_f (drive s j))
    done;
    Gru_f32.forward ();

    (* Dropout on the new hidden state, then carry it as the next state. *)
    for i = 0 to hs - 1 do
      let h = F32.get Gru_f32.h_new i in
      if keep seed k i then
        F32.set Gru_f32.h i h
      else begin
        F32.set Gru_f32.h i 0.0;
        Array.unsafe_set masked_units 0 ((Array.unsafe_get masked_units 0) + 1)
      end
    done;

    (* The loss seed: drive the state towards zero. One squared-error derivative
       per unit, which is 2*h; the 2 is folded into the learning rate. *)
    for i = 0 to hs - 1 do
      F32.set Gru_f32.g_h_new i
        (F32.get Gru_f32.h i)
    done;
    Gru_f32.zero_grads ();
    Gru_f32.backward ();

    (* SGD. Every block, in a fixed order. *)
    let step = of_f lr in
    for i = 0 to (240 * ins) - 1 do
      let p = of_f (F32.get Gru_f32.w_ih i) in
      let g = of_f (F32.get Gru_f32.g_w_ih i) in
      F32.set Gru_f32.w_ih i (to_f (add p (mul step g)))
    done;
    for i = 0 to (240 * hs) - 1 do
      let p = of_f (F32.get Gru_f32.w_hh i) in
      let g = of_f (F32.get Gru_f32.g_w_hh i) in
      F32.set Gru_f32.w_hh i (to_f (add p (mul step g)))
    done;
    for i = 0 to 239 do
      let p = of_f (F32.get Gru_f32.b_ih i) in
      let g = of_f (F32.get Gru_f32.g_b_ih i) in
      F32.set Gru_f32.b_ih i (to_f (add p (mul step g)));
      let q = of_f (F32.get Gru_f32.b_hh i) in
      let d = of_f (F32.get Gru_f32.g_b_hh i) in
      F32.set Gru_f32.b_hh i (to_f (add q (mul step d)))
    done;

    Array.unsafe_set steps_taken 0 ((Array.unsafe_get steps_taken 0) + 1)
  done
