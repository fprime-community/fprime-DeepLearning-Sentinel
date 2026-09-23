(* Section 57: 52's two layers and head, with the HEAD-ONLY loss and the three tolerance
 * models 57.3 derives. gru_deep.ml is NOT edited -- 52's figures were measured against it and
 * an edit would move what they were measured against, which is the rule 52 itself applied to
 * 48's gru_cell.ml. This file is that file plus two magnitude tapes and two accumulators.
 *
 *   s_out   M-51    sum_j |coeff_y[j] * y[j]|, the cancelled head total. 51's, unchanged.
 *   s_flat  M-flat  the head written out: sum_j |c_j| (|b_j| + sum_k |w_jk h_k|).
 *   s_rec   M-rec   s_flat plus the recurrence's accumulated magnitude, weighted by
 *                   |dL/dh| at each step. This is where T re-enters.
 *
 * Originally: Section 52: two GRU layers and the output head, forward and BPTT, under
 * [@zero_alloc strict]. gru_cell.ml and gru_seq.ml are NOT edited or reused; this module
 * carries the same algebra parameterised by layer, so that one copy serves both.
 *
 * (!) 50.2's failure modes are carried forward: the tapes are FLAT float arrays with
 * computed indices, and no helper returns a float -- the loss and the accumulated magnitude
 * are written into one-element arrays.
 *
 * The algebra is flight/include/sentinel/Gru.hpp:22-40, with b_hn INSIDE the reset product
 * and ATen's (h - n) * z + n, exactly as 48.1 transcribed it. *)


(* D82: every element access below is bounds-checked. `acc.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and all 56 `[@zero_alloc strict]` sites still
   hold. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc

let hs = 80                       (* Config.hpp MAX_HIDDEN *)
let ins = 16                      (* Config.hpp MAX_INPUTS *)
let gw = 240                      (* MAX_GATE_WIDTH *)
let n_out = 160                   (* MAX_OUTPUTS = MAX_PREDICTIONS * MAX_CHANNELS *)
let t_max = 250                   (* the flown window, lstm.py:90 *)
let tw = 5                        (* r, z, n, rec_n, h_in -- 50.8's figure of record *)

(* -- layer 0 ---------------------------------------------------------------------------- *)
let w_ih0 = Array.make (gw * ins) 0.0
let w_hh0 = Array.make (gw * hs) 0.0
let b_ih0 = Array.make gw 0.0
let b_hh0 = Array.make gw 0.0
let g_w_ih0 = Array.make (gw * ins) 0.0
let g_w_hh0 = Array.make (gw * hs) 0.0
let g_b_ih0 = Array.make gw 0.0
let g_b_hh0 = Array.make gw 0.0

(* -- layer 1 ---------------------------------------------------------------------------- *)
let w_ih1 = Array.make (gw * hs) 0.0
let w_hh1 = Array.make (gw * hs) 0.0
let b_ih1 = Array.make gw 0.0
let b_hh1 = Array.make gw 0.0
let g_w_ih1 = Array.make (gw * hs) 0.0
let g_w_hh1 = Array.make (gw * hs) 0.0
let g_b_ih1 = Array.make gw 0.0
let g_b_hh1 = Array.make gw 0.0

(* -- head ------------------------------------------------------------------------------- *)
let head_w = Array.make (n_out * hs) 0.0
let head_b = Array.make n_out 0.0
let g_head_w = Array.make (n_out * hs) 0.0
let g_head_b = Array.make n_out 0.0

(* -- state, inputs, tapes ---------------------------------------------------------------- *)
let h_init0 = Array.make hs 0.0
let h_init1 = Array.make hs 0.0
let g_h_init0 = Array.make hs 0.0
let g_h_init1 = Array.make hs 0.0
let x_seq = Array.make (t_max * ins) 0.0
let g_x_seq = Array.make (t_max * ins) 0.0
let out0 = Array.make (t_max * hs) 0.0          (* layer 0's output = layer 1's input *)
let tape0 = Array.make (t_max * tw * hs) 0.0
let tape1 = Array.make (t_max * tw * hs) 0.0
let h_final1 = Array.make hs 0.0                (* layer 1's last output, the head's input *)
let y = Array.make n_out 0.0
let coeff_h = Array.make (t_max * hs) 0.0
let coeff_y = Array.make n_out 0.0
let loss_out = Array.make 1 0.0
let s_out = Array.make 1 0.0                    (* 51.1's accumulated magnitude *)
let s_flat = Array.make 1 0.0                   (* 57.3 M-flat *)
let s_rec = Array.make 1 0.0                    (* 57.3 M-rec *)

(* Per-step, per-unit accumulated absolute magnitude, one tape per layer. Claimed
   once at module level, which is CPP-1's shape and 50.1's. *)
let mag0 = Array.make (t_max * hs) 0.0
let mag1 = Array.make (t_max * hs) 0.0
let mag_p = Array.make gw 0.0
let mag_r = Array.make gw 0.0

(* -- scratch, claimed once ---------------------------------------------------------------- *)
let proj = Array.make gw 0.0
let rec_ = Array.make gw 0.0
let carry0 = Array.make hs 0.0
let carry1 = Array.make hs 0.0
let g_in0 = Array.make ins 0.0
let g_in1 = Array.make hs 0.0
let g_hin = Array.make hs 0.0
let g_hout = Array.make hs 0.0

let[@inline] sigmoid v =
  if v >= 0.0 then 1.0 /. (1.0 +. exp (-. v))
  else let e = exp v in e /. (1.0 +. e)

(* -- one layer, one step, forward --------------------------------------------------------- *)
let[@zero_alloc strict] layer_fwd w_ih w_hh b_ih b_hh n_in in_arr in_off h_arr h_off
                                 tape tbase out_arr out_off magt mbase =
  for gi = 0 to gw - 1 do
    let acc = ref (Array.unsafe_get b_ih gi) in
    let am = ref (Float.abs (Array.unsafe_get b_ih gi)) in
    for j = 0 to n_in - 1 do
      let term = (Array.unsafe_get w_ih ((gi * n_in) + j))
                 *. (Array.unsafe_get in_arr (in_off + j)) in
      acc := !acc +. term;
      am := !am +. (Float.abs term)
    done;
    Array.unsafe_set proj gi !acc;
    Array.unsafe_set mag_p gi !am
  done;
  for gi = 0 to gw - 1 do
    let acc = ref (Array.unsafe_get b_hh gi) in
    let am = ref (Float.abs (Array.unsafe_get b_hh gi)) in
    for j = 0 to hs - 1 do
      let term = (Array.unsafe_get w_hh ((gi * hs) + j))
                 *. (Array.unsafe_get h_arr (h_off + j)) in
      acc := !acc +. term;
      am := !am +. (Float.abs term)
    done;
    Array.unsafe_set rec_ gi !acc;
    Array.unsafe_set mag_r gi !am
  done;
  for i = 0 to hs - 1 do
    let ri = sigmoid ((Array.unsafe_get proj i) +. (Array.unsafe_get rec_ i)) in
    let zi = sigmoid ((Array.unsafe_get proj (hs + i)) +. (Array.unsafe_get rec_ (hs + i))) in
    let rn = Array.unsafe_get rec_ ((2 * hs) + i) in
    let ni = tanh ((Array.unsafe_get proj ((2 * hs) + i)) +. (ri *. rn)) in
    let hi = Array.unsafe_get h_arr (h_off + i) in
    Array.unsafe_set tape (tbase + i) ri;
    Array.unsafe_set tape (tbase + hs + i) zi;
    Array.unsafe_set tape (tbase + (2 * hs) + i) ni;
    Array.unsafe_set tape (tbase + (3 * hs) + i) rn;
    Array.unsafe_set tape (tbase + (4 * hs) + i) hi;
    Array.unsafe_set out_arr (out_off + i) (((hi -. ni) *. zi) +. ni);
    (* 57.3: the absolute magnitude accumulated in producing this unit -- all three
       gates, both affines. The state update itself adds |hi|, |ni| and their product. *)
    Array.unsafe_set magt (mbase + i)
      ((Array.unsafe_get mag_p i) +. (Array.unsafe_get mag_r i)
       +. (Array.unsafe_get mag_p (hs + i)) +. (Array.unsafe_get mag_r (hs + i))
       +. (Array.unsafe_get mag_p ((2 * hs) + i)) +. (Array.unsafe_get mag_r ((2 * hs) + i))
       +. (Float.abs hi) +. (Float.abs ni))
  done

(* -- one layer, one step, backward --------------------------------------------------------
   Reads g_hout (dL/dh_out at this step). Writes g_in (dL/d input) and g_hin (dL/dh_in),
   both overwritten rather than accumulated. Accumulates into the weight gradients. *)
let[@zero_alloc strict] layer_bwd w_ih w_hh n_in in_arr in_off tape tbase
                                 g_w_ih g_w_hh g_b_ih g_b_hh g_in =
  for j = 0 to n_in - 1 do Array.unsafe_set g_in j 0.0 done;
  for j = 0 to hs - 1 do Array.unsafe_set g_hin j 0.0 done;
  for i = 0 to hs - 1 do
    let gh = Array.unsafe_get g_hout i in
    let ri = Array.unsafe_get tape (tbase + i) in
    let zi = Array.unsafe_get tape (tbase + hs + i) in
    let ni = Array.unsafe_get tape (tbase + (2 * hs) + i) in
    let rn = Array.unsafe_get tape (tbase + (3 * hs) + i) in
    let hi = Array.unsafe_get tape (tbase + (4 * hs) + i) in

    let d_z = gh *. (hi -. ni) in
    let d_n = gh *. (1.0 -. zi) in
    let d_h_direct = gh *. zi in
    let d_pre_n = d_n *. (1.0 -. (ni *. ni)) in
    let d_r = d_pre_n *. rn in
    let d_rec_n = d_pre_n *. ri in
    let d_pre_z = d_z *. zi *. (1.0 -. zi) in
    let d_pre_r = d_r *. ri *. (1.0 -. ri) in

    Array.unsafe_set g_b_ih i ((Array.unsafe_get g_b_ih i) +. d_pre_r);
    Array.unsafe_set g_b_hh i ((Array.unsafe_get g_b_hh i) +. d_pre_r);
    Array.unsafe_set g_b_ih (hs + i) ((Array.unsafe_get g_b_ih (hs + i)) +. d_pre_z);
    Array.unsafe_set g_b_hh (hs + i) ((Array.unsafe_get g_b_hh (hs + i)) +. d_pre_z);
    Array.unsafe_set g_b_ih ((2 * hs) + i)
      ((Array.unsafe_get g_b_ih ((2 * hs) + i)) +. d_pre_n);
    Array.unsafe_set g_b_hh ((2 * hs) + i)
      ((Array.unsafe_get g_b_hh ((2 * hs) + i)) +. d_rec_n);

    for j = 0 to n_in - 1 do
      let xj = Array.unsafe_get in_arr (in_off + j) in
      let ir = (i * n_in) + j in
      let iz = ((hs + i) * n_in) + j in
      let inn = (((2 * hs) + i) * n_in) + j in
      Array.unsafe_set g_w_ih ir ((Array.unsafe_get g_w_ih ir) +. (d_pre_r *. xj));
      Array.unsafe_set g_w_ih iz ((Array.unsafe_get g_w_ih iz) +. (d_pre_z *. xj));
      Array.unsafe_set g_w_ih inn ((Array.unsafe_get g_w_ih inn) +. (d_pre_n *. xj));
      Array.unsafe_set g_in j
        ((Array.unsafe_get g_in j)
         +. (d_pre_r *. (Array.unsafe_get w_ih ir))
         +. (d_pre_z *. (Array.unsafe_get w_ih iz))
         +. (d_pre_n *. (Array.unsafe_get w_ih inn)))
    done;
    for j = 0 to hs - 1 do
      let hj = Array.unsafe_get tape (tbase + (4 * hs) + j) in
      let ir = (i * hs) + j in
      let iz = ((hs + i) * hs) + j in
      let inn = (((2 * hs) + i) * hs) + j in
      Array.unsafe_set g_w_hh ir ((Array.unsafe_get g_w_hh ir) +. (d_pre_r *. hj));
      Array.unsafe_set g_w_hh iz ((Array.unsafe_get g_w_hh iz) +. (d_pre_z *. hj));
      Array.unsafe_set g_w_hh inn ((Array.unsafe_get g_w_hh inn) +. (d_rec_n *. hj));
      Array.unsafe_set g_hin j
        ((Array.unsafe_get g_hin j)
         +. (d_pre_r *. (Array.unsafe_get w_hh ir))
         +. (d_pre_z *. (Array.unsafe_get w_hh iz))
         +. (d_rec_n *. (Array.unsafe_get w_hh inn)))
    done;
    Array.unsafe_set g_hin i ((Array.unsafe_get g_hin i) +. d_h_direct)
  done

let h_cur1 = Array.make hs 0.0    (* layer 1's running state; aliased as in and out *)

(* -- the whole model, forward -------------------------------------------------------------- *)
let[@zero_alloc strict] forward_deep t_steps =
  Array.unsafe_set loss_out 0 0.0;
  Array.unsafe_set s_out 0 0.0;
  Array.unsafe_set s_flat 0 0.0;
  for i = 0 to hs - 1 do Array.unsafe_set h_cur1 i (Array.unsafe_get h_init1 i) done;
  for t = 0 to t_steps - 1 do
    let tb = t * tw * hs in
    (* layer 0: input is x_t, state is its own previous output *)
    if t = 0 then
      layer_fwd w_ih0 w_hh0 b_ih0 b_hh0 ins x_seq (t * ins) h_init0 0
                tape0 tb out0 (t * hs) mag0 (t * hs)
    else
      layer_fwd w_ih0 w_hh0 b_ih0 b_hh0 ins x_seq (t * ins) out0 ((t - 1) * hs)
                tape0 tb out0 (t * hs) mag0 (t * hs);
    (* layer 1: input is layer 0's output at this step; state is h_cur1, written in place *)
    layer_fwd w_ih1 w_hh1 b_ih1 b_hh1 hs out0 (t * hs) h_cur1 0 tape1 tb h_cur1 0
              mag1 (t * hs);
    for i = 0 to hs - 1 do
      let term = (Array.unsafe_get coeff_h ((t * hs) + i)) *. (Array.unsafe_get h_cur1 i) in
      Array.unsafe_set loss_out 0 ((Array.unsafe_get loss_out 0) +. term);
      Array.unsafe_set s_out 0 ((Array.unsafe_get s_out 0) +. (Float.abs term))
    done
  done;
  for i = 0 to hs - 1 do Array.unsafe_set h_final1 i (Array.unsafe_get h_cur1 i) done;
  (* the head: y = head_w @ h_last + head_b *)
  for i = 0 to n_out - 1 do
    let acc = ref (Array.unsafe_get head_b i) in
    let am = ref (Float.abs (Array.unsafe_get head_b i)) in
    for j = 0 to hs - 1 do
      let term = (Array.unsafe_get head_w ((i * hs) + j))
                 *. (Array.unsafe_get h_final1 j) in
      acc := !acc +. term;
      am := !am +. (Float.abs term)
    done;
    Array.unsafe_set y i !acc;
    let term = (Array.unsafe_get coeff_y i) *. !acc in
    Array.unsafe_set loss_out 0 ((Array.unsafe_get loss_out 0) +. term);
    (* M-51: the cancelled total. M-flat: the head written out. 57.3. *)
    Array.unsafe_set s_out 0 ((Array.unsafe_get s_out 0) +. (Float.abs term));
    Array.unsafe_set s_flat 0
      ((Array.unsafe_get s_flat 0)
       +. ((Float.abs (Array.unsafe_get coeff_y i)) *. !am))
  done

let[@zero_alloc strict] zero_grads_deep () =
  for i = 0 to (gw * ins) - 1 do Array.unsafe_set g_w_ih0 i 0.0 done;
  for i = 0 to (gw * hs) - 1 do
    Array.unsafe_set g_w_hh0 i 0.0;
    Array.unsafe_set g_w_ih1 i 0.0;
    Array.unsafe_set g_w_hh1 i 0.0
  done;
  for i = 0 to gw - 1 do
    Array.unsafe_set g_b_ih0 i 0.0; Array.unsafe_set g_b_hh0 i 0.0;
    Array.unsafe_set g_b_ih1 i 0.0; Array.unsafe_set g_b_hh1 i 0.0
  done;
  for i = 0 to (n_out * hs) - 1 do Array.unsafe_set g_head_w i 0.0 done;
  for i = 0 to n_out - 1 do Array.unsafe_set g_head_b i 0.0 done;
  for i = 0 to (t_max * ins) - 1 do Array.unsafe_set g_x_seq i 0.0 done;
  for i = 0 to hs - 1 do
    Array.unsafe_set g_h_init0 i 0.0; Array.unsafe_set g_h_init1 i 0.0;
    Array.unsafe_set carry0 i 0.0; Array.unsafe_set carry1 i 0.0
  done

(* -- the whole model, backward through time and down through layers ------------------------- *)
let[@zero_alloc strict] backward_deep t_steps =
  zero_grads_deep ();
  Array.unsafe_set s_rec 0 (Array.unsafe_get s_flat 0);
  (* the head first: its gradient enters layer 1 at the last step only *)
  for i = 0 to n_out - 1 do
    let gy = Array.unsafe_get coeff_y i in
    Array.unsafe_set g_head_b i ((Array.unsafe_get g_head_b i) +. gy);
    for j = 0 to hs - 1 do
      let idx = (i * hs) + j in
      Array.unsafe_set g_head_w idx
        ((Array.unsafe_get g_head_w idx) +. (gy *. (Array.unsafe_get h_final1 j)));
      Array.unsafe_set carry1 j
        ((Array.unsafe_get carry1 j) +. (gy *. (Array.unsafe_get head_w idx)))
    done
  done;
  for t = t_steps - 1 downto 0 do
    let tb = t * tw * hs in
    (* layer 1 *)
    for i = 0 to hs - 1 do
      Array.unsafe_set g_hout i
        ((Array.unsafe_get coeff_h ((t * hs) + i)) +. (Array.unsafe_get carry1 i));
      (* M-rec: |dL/dh1[t,i]| x the magnitude accumulated in producing it. 57.3. *)
      Array.unsafe_set s_rec 0
        ((Array.unsafe_get s_rec 0)
         +. ((Float.abs (Array.unsafe_get g_hout i))
             *. (Array.unsafe_get mag1 ((t * hs) + i))))
    done;
    layer_bwd w_ih1 w_hh1 hs out0 (t * hs) tape1 tb
              g_w_ih1 g_w_hh1 g_b_ih1 g_b_hh1 g_in1;
    for i = 0 to hs - 1 do Array.unsafe_set carry1 i (Array.unsafe_get g_hin i) done;
    (* layer 0: its incoming gradient is layer 1's input gradient plus its own future *)
    for i = 0 to hs - 1 do
      Array.unsafe_set g_hout i
        ((Array.unsafe_get g_in1 i) +. (Array.unsafe_get carry0 i));
      Array.unsafe_set s_rec 0
        ((Array.unsafe_get s_rec 0)
         +. ((Float.abs (Array.unsafe_get g_hout i))
             *. (Array.unsafe_get mag0 ((t * hs) + i))))
    done;
    layer_bwd w_ih0 w_hh0 ins x_seq (t * ins) tape0 tb
              g_w_ih0 g_w_hh0 g_b_ih0 g_b_hh0 g_in0;
    for j = 0 to ins - 1 do
      Array.unsafe_set g_x_seq ((t * ins) + j) (Array.unsafe_get g_in0 j)
    done;
    for i = 0 to hs - 1 do Array.unsafe_set carry0 i (Array.unsafe_get g_hin i) done
  done;
  for i = 0 to hs - 1 do
    Array.unsafe_set g_h_init1 i (Array.unsafe_get carry1 i);
    Array.unsafe_set g_h_init0 i (Array.unsafe_get carry0 i)
  done
