(* Section 64, EX1: 60's F32 gradients at EVERY index, not a sample.
 *
 * 60.6a records that neither of AS4's sampled worst ratios is a bound: strides 101 and 1021
 * are coprime, so their index sets agree only at 0, and the SPARSER sample returned the
 * WORSE ratio. 60.7 owed an exhaustive check "if any later rung makes the 5.3x margin
 * matter", and that condition is met.
 *
 * (!) `deep_f32_check.ml` IS NOT MODIFIED AND THIS IS WHY THIS FILE EXISTS. That module is
 * 60.6's cited artifact; changing it to add a shard offset would edit the instrument the
 * recorded figures were measured on. This module reuses `Deep_f32` unchanged and duplicates
 * nothing but the constants, which are quoted from it below with their source line.
 *
 * (!) STOP 39: k_const, macheps32, h_rel and rtol are 49.2's and 60's, unchanged. None of
 * them moves after a number is seen.
 *
 * Sharding: shard k of n checks indices k, k+n, k+2n, ... Every index is covered exactly
 * once across the n shards, because each index i belongs to shard i mod n and to no other. *)

(* deep_f32_check.ml:23-27, verbatim. *)
let macheps32 = 5.9604645e-8                     (* 2^-24 *)
let h_rel = Float.cbrt macheps32
let rtol = 1e-6
let k_const = 4.0
let step_of pv = h_rel *. Float.max (Float.abs pv) 1.0

(* deep_f32_check.ml:8-20, verbatim: the LCG and the seed. *)
let fill (a : Deep_f32.f32) scale off =
  let s = ref (12345 + off) in
  for i = 0 to Bigarray.Array1.dim a - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    Bigarray.Array1.unsafe_set a i
      (scale *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0))
  done

let seed_model () =
  fill Deep_f32.p 0.1 1;
  fill Deep_f32.h_init0 0.1 5; fill Deep_f32.h_init1 0.1 6;
  fill Deep_f32.x_seq 1.0 8;
  fill Deep_f32.coeff_y 1.0 17

(* deep_f32_check.ml:29-37, verbatim. *)
let loss t = Deep_f32.forward_deep t; Deep_f32.loss_out.(0)

let central_fd idx h t =
  let saved = Bigarray.Array1.unsafe_get Deep_f32.p idx in
  Bigarray.Array1.unsafe_set Deep_f32.p idx (saved +. h);
  let lp = loss t in
  Bigarray.Array1.unsafe_set Deep_f32.p idx (saved -. h);
  let lm = loss t in
  Bigarray.Array1.unsafe_set Deep_f32.p idx saved;
  (lp -. lm) /. (2.0 *. h)

let () =
  let shard  = int_of_string Sys.argv.(1) in
  let shards = int_of_string Sys.argv.(2) in
  let t      = if Array.length Sys.argv > 3 then int_of_string Sys.argv.(3) else 250 in

  seed_model ();
  Deep_f32.forward_deep t;
  Deep_f32.backward_deep t;
  let s_mag = Deep_f32.s_out.(0) in
  let atol = k_const *. macheps32 *. s_mag /. h_rel in

  let ga = Array.make Deep_f32.n_params 0.0 in
  for i = 0 to Deep_f32.n_params - 1 do
    ga.(i) <- Bigarray.Array1.unsafe_get Deep_f32.g i done;

  let out = ref 0 and checked = ref 0 in
  let worst = ref 0.0 and worst_ratio = ref 0.0 and worst_idx = ref (-1) in
  let i = ref shard in
  while !i < Deep_f32.n_params do
    let pv = Bigarray.Array1.unsafe_get Deep_f32.p !i in
    let h = step_of pv in
    let fd = central_fd !i h t in
    let err = Float.abs (fd -. ga.(!i)) in
    let allowed = atol +. (rtol *. Float.abs ga.(!i)) in
    if err > allowed then incr out;
    if err > !worst then worst := err;
    if err /. allowed > !worst_ratio then begin
      worst_ratio := err /. allowed; worst_idx := !i end;
    incr checked;
    i := !i + shards
  done;
  (* one machine-readable line per shard; the driver aggregates *)
  Printf.printf "SHARD %d %d checked=%d outside=%d worst=%.9e ratio=%.6f idx=%d S=%.6e atol=%.6e\n"
    shard shards !checked !out !worst !worst_ratio !worst_idx s_mag atol
