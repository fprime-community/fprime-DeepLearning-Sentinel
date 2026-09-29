(* Section 53: Adam under [@zero_alloc strict], transcribed from torch/optim/adam.py.
 *
 * (!) 53.2's THREE DETAILS, and a transcription that "tidied" any of them would be wrong:
 *   1. the first moment is a LERP          adam.py:456   m += (1-b1)*(g-m)
 *   2. the second moment is NOT            adam.py:475   v = v*b2 + (1-b2)*g*g
 *   3. the bias correction divides roots   adam.py:530-546
 *                                          denom = sqrt(v)/ (1-b2^t)**0.5 + eps
 *                                          p += (-step_size) * m/denom
 *
 * (!) 53.3: ( ** ) and sqrt are stdlib externals carrying [@@unboxed] [@@noalloc]
 * (stdlib.mli:479-480, :485-486). Array.make carries no such declaration, which is why 48's
 * G5 rejected it and why AD2 predicts these hold. *)


(* D82: every element access below is bounds-checked. `acc.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and every `[@zero_alloc strict]` site still
   holds. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc

let n_max = 75360                 (* 52's model, ModelFile.hpp:34 at Config.hpp's maxima *)

let lr = 1e-3                     (* lstm.py:121 *)
let beta1 = 0.9                   (* adam.py:39 *)
let beta2 = 0.999                 (* adam.py:39 *)
let eps = 1e-8                    (* adam.py:40 *)

let m = Array.make n_max 0.0
let v = Array.make n_max 0.0

(* one-element slots rather than returns: 48's G1 showed a float return boxes *)
let t_step = Array.make 1 0.0
let step_size = Array.make 1 0.0
let bc2_sqrt = Array.make 1 0.0
let b1_running = Array.make 1 1.0
let b2_running = Array.make 1 1.0

let[@zero_alloc strict] reset () =
  for i = 0 to n_max - 1 do
    Array.unsafe_set m i 0.0; Array.unsafe_set v i 0.0
  done;
  Array.unsafe_set t_step 0 0.0;
  Array.unsafe_set b1_running 0 1.0;
  Array.unsafe_set b2_running 0 1.0

(* AD2 form A: the bias correction by ( ** ), which is what torch writes *)
let[@zero_alloc strict] begin_step_pow () =
  Array.unsafe_set t_step 0 ((Array.unsafe_get t_step 0) +. 1.0);
  let t = Array.unsafe_get t_step 0 in
  let bc1 = 1.0 -. (beta1 ** t) in
  let bc2 = 1.0 -. (beta2 ** t) in
  Array.unsafe_set step_size 0 (lr /. bc1);
  Array.unsafe_set bc2_sqrt 0 (bc2 ** 0.5)

(* AD2 form B: the same correction by a running product, no call to ( ** ) at all *)
let[@zero_alloc strict] begin_step_running () =
  Array.unsafe_set t_step 0 ((Array.unsafe_get t_step 0) +. 1.0);
  Array.unsafe_set b1_running 0 ((Array.unsafe_get b1_running 0) *. beta1);
  Array.unsafe_set b2_running 0 ((Array.unsafe_get b2_running 0) *. beta2);
  let bc1 = 1.0 -. (Array.unsafe_get b1_running 0) in
  let bc2 = 1.0 -. (Array.unsafe_get b2_running 0) in
  Array.unsafe_set step_size 0 (lr /. bc1);
  Array.unsafe_set bc2_sqrt 0 (sqrt bc2)

(* One parameter block. `moff` is its base in the shared m/v index space. *)
let[@zero_alloc strict] update_block p g moff n =
  let ss = Array.unsafe_get step_size 0 in
  let b2s = Array.unsafe_get bc2_sqrt 0 in
  for i = 0 to n - 1 do
    let gi = Array.unsafe_get g i in
    let mi = Array.unsafe_get m (moff + i) in
    (* detail 1: lerp, not a weighted sum *)
    let mi' = mi +. ((1.0 -. beta1) *. (gi -. mi)) in
    Array.unsafe_set m (moff + i) mi';
    (* detail 2: mul then addcmul, which IS the textbook form *)
    let vi' = ((Array.unsafe_get v (moff + i)) *. beta2)
              +. ((1.0 -. beta2) *. gi *. gi) in
    Array.unsafe_set v (moff + i) vi';
    (* detail 3: two roots divided, eps added after *)
    let denom = ((sqrt vi') /. b2s) +. eps in
    Array.unsafe_set p i ((Array.unsafe_get p i) +. ((-. ss) *. (mi' /. denom)))
  done
