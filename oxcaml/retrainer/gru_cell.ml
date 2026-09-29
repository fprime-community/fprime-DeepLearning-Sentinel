(* Section 48: one GRU cell, forward and backward, under [@zero_alloc strict].
 *
 * The algebra is the FLOWN cell's, transcribed from flight/include/sentinel/Gru.hpp:22-40:
 *
 *     r  = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
 *     z  = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
 *     n  = tanh   (W_in x + b_in + r * (W_hn h + b_hn))
 *     h' = (h - n) * z + n
 *
 * (!) TWO DETAILS A TEXTBOOK GRU GETS WRONG AND THIS ONE MUST NOT.
 *   1. b_hn sits INSIDE the reset product. The recurrent product is formed once for all
 *      three gates with the FULL b_hh added, and the third block is then multiplied by r.
 *      A loader folding b_hh into b_ih would be wrong on exactly that gate (D26).
 *   2. The update is ATen's (h - n) * z + n, NOT the textbook (1 - z) n + z h. Equal in
 *      exact arithmetic, not in floating point, and every measurement in this project was
 *      taken against ATen's.
 *
 * Gate block order is r, z, n -- rows 0..H-1, H..2H-1, 2H..3H-1 of the 3H-row matrices.
 *
 * Everything is preallocated at module initialisation and nothing below allocates: that is
 * CPP-1's shape and it is what 48.4's G1 and G2 assert. No `assume` appears in this file. *)


(* D82: every element access below is bounds-checked. `acc.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and every `[@zero_alloc strict]` site still
   holds. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc

let h_size = 80          (* Config.hpp MAX_HIDDEN *)
let in_size = 16         (* Config.hpp MAX_INPUTS *)
let gates = 3            (* Config.hpp N_GATES *)
let gw = gates * h_size  (* MAX_GATE_WIDTH = 240 *)

(* -- parameters, preallocated ------------------------------------------------------- *)
let w_ih = Array.make (gw * in_size) 0.0
let w_hh = Array.make (gw * h_size) 0.0
let b_ih = Array.make gw 0.0
let b_hh = Array.make gw 0.0
let x = Array.make in_size 0.0
let h = Array.make h_size 0.0

(* -- gradients, preallocated, accumulated into ---------------------------------------- *)
let g_w_ih = Array.make (gw * in_size) 0.0
let g_w_hh = Array.make (gw * h_size) 0.0
let g_b_ih = Array.make gw 0.0
let g_b_hh = Array.make gw 0.0
let g_x = Array.make in_size 0.0
let g_h = Array.make h_size 0.0

(* -- intermediate activations the backward pass needs, preallocated ------------------- *)
let proj = Array.make gw 0.0      (* W_ih x + b_ih, all three gates *)
let rec_ = Array.make gw 0.0      (* W_hh h + b_hh, all three gates *)
let r = Array.make h_size 0.0
let z = Array.make h_size 0.0
let n = Array.make h_size 0.0
let h_new = Array.make h_size 0.0
let g_h_new = Array.make h_size 0.0   (* the incoming gradient dL/dh' *)

(* Logistic in the branch form Gru.hpp uses, which does not overflow for large |v|.
 *
 * (!) [@inline] IS LOAD-BEARING AND IS SECTION 48's ONE ACCOMMODATION. Measured, not
 * guessed: with no annotation and no hint on this function, `forward` FAILS strict with
 * "called function may allocate (direct call camlGru_cell__sigmoid)" -- a float RETURNED
 * across a function boundary is boxed, the same 16 bytes 47.13.4 hit on `checksum`, and
 * flambda2 does NOT inline it unasked. With [@inline] the caller is clean.
 *
 * It is a hint, so 48.4's G1 band gives NO VERDICT rather than HOLD, and it is itemised
 * here as that band requires. The annotation is dropped from this function because once
 * inlined it has no independent body to check; the property is proved where it matters,
 * on `forward`.
 *
 * (!) AND `backward` NEEDS NONE OF THIS. It passes strict with no hint and no assume,
 * because it never calls a float-returning helper -- it reads the stored r, z and n and
 * differentiates them in place. That asymmetry is 48's result, not an aside. *)
let[@inline] sigmoid v =
  if v >= 0.0 then 1.0 /. (1.0 +. exp (-. v))
  else
    let e = exp v in
    e /. (1.0 +. e)

(* -- forward ------------------------------------------------------------------------- *)
let[@zero_alloc strict] forward () =
  (* projected = W_ih x + b_ih, for all three gate blocks at once *)
  for g = 0 to gw - 1 do
    let acc = ref (Array.unsafe_get b_ih g) in
    for j = 0 to in_size - 1 do
      acc := !acc +. (Array.unsafe_get w_ih ((g * in_size) + j))
                     *. (Array.unsafe_get x j)
    done;
    Array.unsafe_set proj g !acc
  done;
  (* recurrent = W_hh h + b_hh, with the FULL b_hh added here -- detail 1 above *)
  for g = 0 to gw - 1 do
    let acc = ref (Array.unsafe_get b_hh g) in
    for j = 0 to h_size - 1 do
      acc := !acc +. (Array.unsafe_get w_hh ((g * h_size) + j))
                     *. (Array.unsafe_get h j)
    done;
    Array.unsafe_set rec_ g !acc
  done;
  for i = 0 to h_size - 1 do
    let ri = sigmoid ((Array.unsafe_get proj i) +. (Array.unsafe_get rec_ i)) in
    let zi = sigmoid ((Array.unsafe_get proj (h_size + i))
                      +. (Array.unsafe_get rec_ (h_size + i))) in
    (* detail 1: r multiplies the recurrent block that ALREADY contains b_hn *)
    let ni = tanh ((Array.unsafe_get proj ((2 * h_size) + i))
                   +. (ri *. (Array.unsafe_get rec_ ((2 * h_size) + i)))) in
    Array.unsafe_set r i ri;
    Array.unsafe_set z i zi;
    Array.unsafe_set n i ni;
    (* detail 2: ATen's form *)
    Array.unsafe_set h_new i ((((Array.unsafe_get h i) -. ni) *. zi) +. ni)
  done

(* -- backward ------------------------------------------------------------------------ *)
(* Reads g_h_new (dL/dh'), accumulates into every g_* buffer. Zeroing is the caller's, so
 * that accumulation over a sequence is possible without this function allocating or
 * branching on a flag. *)
let[@zero_alloc strict] backward () =
  for i = 0 to h_size - 1 do
    let gh = Array.unsafe_get g_h_new i in
    let hi = Array.unsafe_get h i in
    let ri = Array.unsafe_get r i in
    let zi = Array.unsafe_get z i in
    let ni = Array.unsafe_get n i in
    let rec_n = Array.unsafe_get rec_ ((2 * h_size) + i) in

    (* h' = (h - n) z + n
       dh'/dz = h - n ;  dh'/dn = 1 - z ;  dh'/dh = z *)
    let d_z = gh *. (hi -. ni) in
    let d_n = gh *. (1.0 -. zi) in
    let d_h_direct = gh *. zi in

    (* n = tanh(proj_n + r * rec_n) ;  dtanh = 1 - n^2 *)
    let d_pre_n = d_n *. (1.0 -. (ni *. ni)) in
    let d_r = d_pre_n *. rec_n in
    (* the reset gate scales the recurrent block, so that block's grad carries r *)
    let d_rec_n = d_pre_n *. ri in

    (* sigmoid' = s (1 - s) *)
    let d_pre_z = d_z *. zi *. (1.0 -. zi) in
    let d_pre_r = d_r *. ri *. (1.0 -. ri) in

    (* proj and rec share the pre-activation for r and z; for n they differ *)
    Array.unsafe_set g_b_ih i ((Array.unsafe_get g_b_ih i) +. d_pre_r);
    Array.unsafe_set g_b_hh i ((Array.unsafe_get g_b_hh i) +. d_pre_r);
    Array.unsafe_set g_b_ih (h_size + i)
      ((Array.unsafe_get g_b_ih (h_size + i)) +. d_pre_z);
    Array.unsafe_set g_b_hh (h_size + i)
      ((Array.unsafe_get g_b_hh (h_size + i)) +. d_pre_z);
    Array.unsafe_set g_b_ih ((2 * h_size) + i)
      ((Array.unsafe_get g_b_ih ((2 * h_size) + i)) +. d_pre_n);
    (* b_hn is inside the reset product, so its gradient carries r -- detail 1 *)
    Array.unsafe_set g_b_hh ((2 * h_size) + i)
      ((Array.unsafe_get g_b_hh ((2 * h_size) + i)) +. d_rec_n);

    (* weights and inputs, for each of the three gate rows *)
    for j = 0 to in_size - 1 do
      let xj = Array.unsafe_get x j in
      let ir = (i * in_size) + j in
      let iz = ((h_size + i) * in_size) + j in
      let inn = (((2 * h_size) + i) * in_size) + j in
      Array.unsafe_set g_w_ih ir ((Array.unsafe_get g_w_ih ir) +. (d_pre_r *. xj));
      Array.unsafe_set g_w_ih iz ((Array.unsafe_get g_w_ih iz) +. (d_pre_z *. xj));
      Array.unsafe_set g_w_ih inn ((Array.unsafe_get g_w_ih inn) +. (d_pre_n *. xj));
      Array.unsafe_set g_x j
        ((Array.unsafe_get g_x j)
         +. (d_pre_r *. (Array.unsafe_get w_ih ir))
         +. (d_pre_z *. (Array.unsafe_get w_ih iz))
         +. (d_pre_n *. (Array.unsafe_get w_ih inn)))
    done;
    for j = 0 to h_size - 1 do
      let hj = Array.unsafe_get h j in
      let ir = (i * h_size) + j in
      let iz = ((h_size + i) * h_size) + j in
      let inn = (((2 * h_size) + i) * h_size) + j in
      Array.unsafe_set g_w_hh ir ((Array.unsafe_get g_w_hh ir) +. (d_pre_r *. hj));
      Array.unsafe_set g_w_hh iz ((Array.unsafe_get g_w_hh iz) +. (d_pre_z *. hj));
      Array.unsafe_set g_w_hh inn ((Array.unsafe_get g_w_hh inn) +. (d_rec_n *. hj));
      Array.unsafe_set g_h j
        ((Array.unsafe_get g_h j)
         +. (d_pre_r *. (Array.unsafe_get w_hh ir))
         +. (d_pre_z *. (Array.unsafe_get w_hh iz))
         +. (d_rec_n *. (Array.unsafe_get w_hh inn)))
    done;
    (* h appears in the update itself as well as through the gates *)
    Array.unsafe_set g_h i ((Array.unsafe_get g_h i) +. d_h_direct)
  done

let[@zero_alloc strict] zero_grads () =
  for i = 0 to (gw * in_size) - 1 do Array.unsafe_set g_w_ih i 0.0 done;
  for i = 0 to (gw * h_size) - 1 do Array.unsafe_set g_w_hh i 0.0 done;
  for i = 0 to gw - 1 do
    Array.unsafe_set g_b_ih i 0.0;
    Array.unsafe_set g_b_hh i 0.0
  done;
  for i = 0 to in_size - 1 do Array.unsafe_set g_x i 0.0 done;
  for i = 0 to h_size - 1 do Array.unsafe_set g_h i 0.0 done
