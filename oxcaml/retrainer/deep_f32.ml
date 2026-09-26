(* Section 60 / E5-b: the training cycle assembled at FLOAT32.
 *
 * (!) 54.2a's PERMITTED SET, EXTENDED BY NOTHING (60.2, stop 31):
 *   %addfloat32  %mulfloat32  %divfloat32  %floatoffloat32  %float32offloat
 *   expf  tanhf   (C externals, [@@unboxed] [@@noalloc])
 * Subtraction is add + an exact sign flip (54.2a), so %subfloat32 is NOT used
 * even though 60.2 confirmed it exists. Nor are %negfloat32 or %absfloat32.
 *
 * (!) THE SQUARE ROOT IS ROUTE 2 AND IT IS EXACT (60.2). No float32 sqrt
 * primitive exists in this switch. float64's 53 significand bits satisfy the
 * classical p' >= 2p + 2 condition for square root against float32's 24
 * (2*24+2 = 50 <= 53), so a float64 sqrt rounded back to float32 IS the
 * correctly-rounded float32 square root. Division has no such guarantee, which
 * is why 54.2a needed %divfloat32 and this needs nothing. Measured over
 * 22,052,527 float32 values against sqrtf: 0 differ. AS2 re-checks it in-process.
 *
 * (!) THE ALGORITHM IS 52's, TRANSCRIBED. gru_deep.ml is not edited and not
 * reused; the shapes, the tape width and the accumulation order are its. *)


(* D82: every element access below is bounds-checked. `acc.ml`
   supplies `Array.unsafe_get` / `unsafe_set` as the CHECKED operations, so the
   call sites keep their spelling and every `[@zero_alloc strict]` site still
   holds. `scripts/oxcaml_checked.sh` is the measurement. *)
open Acc

external to_f : float32 -> float = "%floatoffloat32"
external of_f : float -> float32 = "%float32offloat"
external add : float32 -> float32 -> float32 = "%addfloat32"
external mul : float32 -> float32 -> float32 = "%mulfloat32"
external div : float32 -> float32 -> float32 = "%divfloat32"
external expf : float32 -> float32 = "caml_expf_float32" "expf" [@@unboxed] [@@noalloc]
external tanhf : float32 -> float32 = "caml_tanhf_float32" "tanhf" [@@unboxed] [@@noalloc]

let hs = 80                       (* Config.hpp MAX_HIDDEN *)
let ins = 16                      (* Config.hpp MAX_INPUTS *)
let gw = 240                      (* MAX_GATE_WIDTH *)
let n_out = 160                   (* MAX_OUTPUTS *)
let t_max = 250                   (* lstm.py:90 *)
let tw = 5                        (* 50.8's figure of record *)
let n_params = 75360              (* ModelFile.hpp:34, at Config.hpp's maxima *)

type f32 = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
let mk n : f32 = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n
let[@inline] get (a : f32) i = of_f (F32.get a i)
let[@inline] set (a : f32) i (v : float32) = F32.set a i (to_f v)

let zero = of_f 0.0
let one = of_f 1.0
let minus_one = of_f (-1.0)
let[@inline] sub a b = add a (mul b minus_one)                  (* 54.2a: exact *)
let[@inline] sqrtf v = of_f (sqrt (to_f v))                     (* 60.2 route 2 *)
let[@inline] sigmoid v =                                        (* Gru.cpp:28-34 *)
  if to_f v >= 0.0 then div one (add one (expf (mul v minus_one)))
  else let e = expf v in div e (add one e)

(* -- parameters, one flat block so Adam indexes it once -------------------------------- *)
let p = mk n_params
let g = mk n_params
let m = mk n_params
let v = mk n_params
let best = mk n_params

(* offsets into the flat block, in the order 52 declares them *)
let o_wih0 = 0                          and n_wih0 = gw * ins
let o_whh0 = gw * ins                   and n_whh0 = gw * hs
let o_bih0 = (gw * ins) + (gw * hs)
let o_bhh0 = (gw * ins) + (gw * hs) + gw
let o_wih1 = (gw * ins) + (gw * hs) + (2 * gw)
let o_whh1 = (gw * ins) + (2 * gw * hs) + (2 * gw)
let o_bih1 = (gw * ins) + (3 * gw * hs) + (2 * gw)
let o_bhh1 = (gw * ins) + (3 * gw * hs) + (3 * gw)
let o_hw = (gw * ins) + (3 * gw * hs) + (4 * gw)
let o_hb = (gw * ins) + (3 * gw * hs) + (4 * gw) + (n_out * hs)

(* -- state, inputs, tapes -------------------------------------------------------------- *)
let h_init0 = mk hs
let h_init1 = mk hs
let x_seq = mk (t_max * ins)
let out0 = mk (t_max * hs)
let tape0 = mk (t_max * tw * hs)
let tape1 = mk (t_max * tw * hs)
let h_cur1 = mk hs
let h_final1 = mk hs
let y = mk n_out
let coeff_y = mk n_out
(* D85: the TARGET -- the P x C block of values that follows the input window, in the
   head's own (prediction, channel) order. Until D85 this module had no target: its
   loss was 57's head-only TEST functional, sum(coeff_y * y), a vehicle for proving
   gradients, and the flown cycle therefore learned nothing. The loss is now the
   ground's own objective, lstm.py's MSE against the future block, with coeff_y kept
   as a per-output weight (1.0 in flight; the gradient checks fill it). *)
let tgt = mk n_out
let inv_n = of_f (1.0 /. float_of_int n_out)
let two_inv_n = of_f (2.0 /. float_of_int n_out)

let proj = mk gw
let rec_ = mk gw
let carry0 = mk hs
let carry1 = mk hs
let g_hout = mk hs
let g_hin = mk hs
let g_in0 = mk ins
let g_in1 = mk hs

let loss_out = Array.make 1 0.0
let s_out = Array.make 1 0.0            (* 51.2's S, accumulated at F32 *)
let steps_taken = Array.make 1 0

(* -- one layer, one step, forward ------------------------------------------------------ *)
let[@zero_alloc strict] layer_fwd o_wih o_whh o_bih o_bhh n_in (in_arr : f32) in_off
                                 (h_arr : f32) h_off (tape : f32) tbase
                                 (out_arr : f32) out_off =
  for gi = 0 to gw - 1 do
    let acc = ref (get p (o_bih + gi)) in
    for j = 0 to n_in - 1 do
      acc := add !acc (mul (get p (o_wih + (gi * n_in) + j)) (get in_arr (in_off + j)))
    done;
    set proj gi !acc
  done;
  for gi = 0 to gw - 1 do
    let acc = ref (get p (o_bhh + gi)) in
    for j = 0 to hs - 1 do
      acc := add !acc (mul (get p (o_whh + (gi * hs) + j)) (get h_arr (h_off + j)))
    done;
    set rec_ gi !acc
  done;
  for i = 0 to hs - 1 do
    let ri = sigmoid (add (get proj i) (get rec_ i)) in
    let zi = sigmoid (add (get proj (hs + i)) (get rec_ (hs + i))) in
    let rn = get rec_ ((2 * hs) + i) in
    let ni = tanhf (add (get proj ((2 * hs) + i)) (mul ri rn)) in
    let hi = get h_arr (h_off + i) in
    set tape (tbase + i) ri;
    set tape (tbase + hs + i) zi;
    set tape (tbase + (2 * hs) + i) ni;
    set tape (tbase + (3 * hs) + i) rn;
    set tape (tbase + (4 * hs) + i) hi;
    set out_arr (out_off + i) (add (mul (sub hi ni) zi) ni)
  done

(* -- one layer, one step, backward ----------------------------------------------------- *)
let[@zero_alloc strict] layer_bwd o_wih o_whh n_in (in_arr : f32) in_off
                                 (tape : f32) tbase o_gwih o_gwhh o_gbih o_gbhh
                                 (g_in : f32) =
  for j = 0 to n_in - 1 do set g_in j zero done;
  for j = 0 to hs - 1 do set g_hin j zero done;
  for i = 0 to hs - 1 do
    let gh = get g_hout i in
    let ri = get tape (tbase + i) in
    let zi = get tape (tbase + hs + i) in
    let ni = get tape (tbase + (2 * hs) + i) in
    let rn = get tape (tbase + (3 * hs) + i) in
    let hi = get tape (tbase + (4 * hs) + i) in

    let d_z = mul gh (sub hi ni) in
    let d_n = mul gh (sub one zi) in
    let d_h_direct = mul gh zi in
    let d_pre_n = mul d_n (sub one (mul ni ni)) in
    let d_r = mul d_pre_n rn in
    let d_rec_n = mul d_pre_n ri in
    let d_pre_z = mul (mul d_z zi) (sub one zi) in
    let d_pre_r = mul (mul d_r ri) (sub one ri) in

    set g (o_gbih + i) (add (get g (o_gbih + i)) d_pre_r);
    set g (o_gbhh + i) (add (get g (o_gbhh + i)) d_pre_r);
    set g (o_gbih + hs + i) (add (get g (o_gbih + hs + i)) d_pre_z);
    set g (o_gbhh + hs + i) (add (get g (o_gbhh + hs + i)) d_pre_z);
    set g (o_gbih + (2 * hs) + i) (add (get g (o_gbih + (2 * hs) + i)) d_pre_n);
    set g (o_gbhh + (2 * hs) + i) (add (get g (o_gbhh + (2 * hs) + i)) d_rec_n);

    for j = 0 to n_in - 1 do
      let xj = get in_arr (in_off + j) in
      let ir = (i * n_in) + j in
      let iz = ((hs + i) * n_in) + j in
      let inn = (((2 * hs) + i) * n_in) + j in
      set g (o_gwih + ir) (add (get g (o_gwih + ir)) (mul d_pre_r xj));
      set g (o_gwih + iz) (add (get g (o_gwih + iz)) (mul d_pre_z xj));
      set g (o_gwih + inn) (add (get g (o_gwih + inn)) (mul d_pre_n xj));
      set g_in j (add (get g_in j)
                    (add (mul d_pre_r (get p (o_wih + ir)))
                       (add (mul d_pre_z (get p (o_wih + iz)))
                          (mul d_pre_n (get p (o_wih + inn))))))
    done;
    for j = 0 to hs - 1 do
      let hj = get tape (tbase + (4 * hs) + j) in
      let ir = (i * hs) + j in
      let iz = ((hs + i) * hs) + j in
      let inn = (((2 * hs) + i) * hs) + j in
      set g (o_gwhh + ir) (add (get g (o_gwhh + ir)) (mul d_pre_r hj));
      set g (o_gwhh + iz) (add (get g (o_gwhh + iz)) (mul d_pre_z hj));
      set g (o_gwhh + inn) (add (get g (o_gwhh + inn)) (mul d_rec_n hj));
      set g_hin j (add (get g_hin j)
                     (add (mul d_pre_r (get p (o_whh + ir)))
                        (add (mul d_pre_z (get p (o_whh + iz)))
                           (mul d_rec_n (get p (o_whh + inn))))))
    done;
    set g_hin i (add (get g_hin i) d_h_direct)
  done

(* -- the whole model ------------------------------------------------------------------- *)
let[@zero_alloc strict] forward_deep t_steps =
  Array.unsafe_set loss_out 0 0.0;
  Array.unsafe_set s_out 0 0.0;
  for i = 0 to hs - 1 do set h_cur1 i (get h_init1 i) done;
  for t = 0 to t_steps - 1 do
    let tb = t * tw * hs in
    if t = 0 then
      layer_fwd o_wih0 o_whh0 o_bih0 o_bhh0 ins x_seq (t * ins) h_init0 0 tape0 tb out0 (t * hs)
    else
      layer_fwd o_wih0 o_whh0 o_bih0 o_bhh0 ins x_seq (t * ins) out0 ((t - 1) * hs)
                tape0 tb out0 (t * hs);
    layer_fwd o_wih1 o_whh1 o_bih1 o_bhh1 hs out0 (t * hs) h_cur1 0 tape1 tb h_cur1 0
  done;
  for i = 0 to hs - 1 do set h_final1 i (get h_cur1 i) done;
  (* the head, and 57's head-only loss *)
  for i = 0 to n_out - 1 do
    let acc = ref (get p (o_hb + i)) in
    for j = 0 to hs - 1 do
      acc := add !acc (mul (get p (o_hw + (i * hs) + j)) (get h_final1 j))
    done;
    set y i !acc;
    (* D85: coeff_y * (y - target)^2 / n_out -- lstm.py:668's MSE, per output. *)
    let d = sub !acc (get tgt i) in
    let term = mul (get coeff_y i) (mul (mul d d) inv_n) in
    Array.unsafe_set loss_out 0 ((Array.unsafe_get loss_out 0) +. (to_f term));
    Array.unsafe_set s_out 0 ((Array.unsafe_get s_out 0) +. (Float.abs (to_f term)))
  done

let[@zero_alloc strict] zero_grads () =
  for i = 0 to n_params - 1 do set g i zero done

let[@zero_alloc strict] backward_deep t_steps =
  zero_grads ();
  for j = 0 to hs - 1 do set carry1 j zero; set carry0 j zero done;
  for i = 0 to n_out - 1 do
    (* D85: d loss / d y = coeff_y * 2 (y - target) / n_out. *)
    let gy = mul (get coeff_y i) (mul two_inv_n (sub (get y i) (get tgt i))) in
    set g (o_hb + i) (add (get g (o_hb + i)) gy);
    for j = 0 to hs - 1 do
      let idx = (i * hs) + j in
      set g (o_hw + idx) (add (get g (o_hw + idx)) (mul gy (get h_final1 j)));
      set carry1 j (add (get carry1 j) (mul gy (get p (o_hw + idx))))
    done
  done;
  for t = t_steps - 1 downto 0 do
    let tb = t * tw * hs in
    for i = 0 to hs - 1 do set g_hout i (get carry1 i) done;
    layer_bwd o_wih1 o_whh1 hs out0 (t * hs) tape1 tb o_wih1 o_whh1 o_bih1 o_bhh1 g_in1;
    for i = 0 to hs - 1 do set carry1 i (get g_hin i) done;
    for i = 0 to hs - 1 do set g_hout i (add (get g_in1 i) (get carry0 i)) done;
    layer_bwd o_wih0 o_whh0 ins x_seq (t * ins) tape0 tb o_wih0 o_whh0 o_bih0 o_bhh0 g_in0;
    for i = 0 to hs - 1 do set carry0 i (get g_hin i) done
  done

(* -- Adam at float32, 53.2's three details ---------------------------------------------- *)
let lr = of_f 1e-3                (* lstm.py:121 *)
let beta1 = of_f 0.9
let beta2 = of_f 0.999
let eps = of_f 1e-8
let adam_t = Array.make 1 0.0
let b1r = mk 1
let b2r = mk 1

let[@zero_alloc strict] adam_reset () =
  for i = 0 to n_params - 1 do set m i zero; set v i zero done;
  Array.unsafe_set adam_t 0 0.0;
  set b1r 0 one; set b2r 0 one

let[@zero_alloc strict] adam_step () =
  Array.unsafe_set adam_t 0 ((Array.unsafe_get adam_t 0) +. 1.0);
  set b1r 0 (mul (get b1r 0) beta1);
  set b2r 0 (mul (get b2r 0) beta2);
  let bc1 = sub one (get b1r 0) in
  let bc2 = sub one (get b2r 0) in
  let ss = div lr bc1 in
  let b2s = sqrtf bc2 in
  for i = 0 to n_params - 1 do
    let gi = get g i in
    let mi = get m i in
    (* detail 1: a LERP, adam.py:456 *)
    let mi' = add mi (mul (sub one beta1) (sub gi mi)) in
    set m i mi';
    (* detail 2: mul then addcmul, adam.py:475 *)
    let vi' = add (mul (get v i) beta2) (mul (mul (sub one beta2) gi) gi) in
    set v i vi';
    (* detail 3: two roots divided, eps after *)
    let denom = add (div (sqrtf vi') b2s) eps in
    set p i (add (get p i) (mul (mul ss minus_one) (div mi' denom)))
  done

let[@zero_alloc strict] save_best () =
  for i = 0 to n_params - 1 do set best i (get p i) done

(* -- one retraining cycle: D73's fixed STEP budget -------------------------------------- *)
let[@zero_alloc strict] run_cycle budget t_steps admit =
  Array.unsafe_set steps_taken 0 0;
  if admit = 0 then ()                     (* 56's gate refused the window *)
  else begin
    (* D85: THE OPTIMISER'S STATE PERSISTS ACROSS CYCLES. It was reset here on every
       call, which at a budget of 1 per tick makes every step a FIRST Adam step --
       m = g, v = g^2, update = lr * g / |g| -- i.e. sign descent moving every weight by
       lr per tick regardless of its gradient. The reset now happens where a training
       run begins: the warm start (shadow_c.ml) and the seed (cycle_c.ml). *)
    for _ = 1 to budget do
      forward_deep t_steps;
      backward_deep t_steps;
      adam_step ();
      save_best ();                        (* D73 c.2: bookkeeping, not control flow *)
      Array.unsafe_set steps_taken 0 ((Array.unsafe_get steps_taken 0) + 1)
    done
  end
