(* Section 60's driver: AS2 to AS7. Not annotated and not part of the claim. *)

external sqrtf_c : float32 -> float32 = "caml_sqrtf_ref" "sqrtf" [@@unboxed] [@@noalloc]

let to_f = Deep_f32.to_f
let of_f = Deep_f32.of_f

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

(* 49.2's step rule and K, with macheps taken at the precision in use (60.3 AS4). *)
let macheps32 = 5.9604645e-8                     (* 2^-24 *)
let h_rel = Float.cbrt macheps32
let rtol = 1e-6
let k_const = 4.0
let step_of pv = h_rel *. Float.max (Float.abs pv) 1.0

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
  let mode = if Array.length Sys.argv > 1 then Sys.argv.(1) else "all" in
  let t = if Array.length Sys.argv > 2 then int_of_string Sys.argv.(2) else 250 in

  if mode = "dump" then begin
    (* AS3: weights, inputs and the head output, at full precision, for reference.py *)
    seed_model ();
    Deep_f32.forward_deep t;
    let oc = open_out Sys.argv.(3) in
    (* (!) 52's dump format exactly, so scripts/s52_reference_check.py is reused
       VERBATIM rather than re-implemented. The comparison is then the same one
       HD7 made, against the same reference.forward. *)
    Printf.fprintf oc "dims %d %d %d %d %d\n"
      Deep_f32.hs Deep_f32.ins Deep_f32.gw Deep_f32.n_out t;
    let emit name (a : Deep_f32.f32) off n =
      Printf.fprintf oc "%s %d\n" name n;
      for i = 0 to n - 1 do
        Printf.fprintf oc "%.17g\n" (Bigarray.Array1.unsafe_get a (off + i)) done in
    emit "w_ih0" Deep_f32.p Deep_f32.o_wih0 (Deep_f32.gw * Deep_f32.ins);
    emit "w_hh0" Deep_f32.p Deep_f32.o_whh0 (Deep_f32.gw * Deep_f32.hs);
    emit "b_ih0" Deep_f32.p Deep_f32.o_bih0 Deep_f32.gw;
    emit "b_hh0" Deep_f32.p Deep_f32.o_bhh0 Deep_f32.gw;
    emit "w_ih1" Deep_f32.p Deep_f32.o_wih1 (Deep_f32.gw * Deep_f32.hs);
    emit "w_hh1" Deep_f32.p Deep_f32.o_whh1 (Deep_f32.gw * Deep_f32.hs);
    emit "b_ih1" Deep_f32.p Deep_f32.o_bih1 Deep_f32.gw;
    emit "b_hh1" Deep_f32.p Deep_f32.o_bhh1 Deep_f32.gw;
    emit "head_w" Deep_f32.p Deep_f32.o_hw (Deep_f32.n_out * Deep_f32.hs);
    emit "head_b" Deep_f32.p Deep_f32.o_hb Deep_f32.n_out;
    emit "h_init0" Deep_f32.h_init0 0 Deep_f32.hs;
    emit "h_init1" Deep_f32.h_init1 0 Deep_f32.hs;
    emit "x_seq" Deep_f32.x_seq 0 (t * Deep_f32.ins);
    emit "y" Deep_f32.y 0 Deep_f32.n_out;
    close_out oc;
    Printf.printf "   dumped in 52's format, T = %d -> %s\n" t Sys.argv.(3)
  end else begin
    (* AS2: route 2 against a correctly-rounded reference, in-process. *)
    let bad = ref 0 and n = ref 0 in
    let bits = ref 1 in
    while !bits < 0x7F800000 do
      let x = Int32.float_of_bits (Int32.of_int (!bits)) in
      let a = sqrtf_c (of_f x) and b = Deep_f32.sqrtf (of_f x) in
      if Int32.bits_of_float (to_f a) <> Int32.bits_of_float (to_f b) then incr bad;
      incr n; bits := !bits + 7919
    done;
    Printf.printf "   AS2  route 2 vs sqrtf over %d float32 values: %d differ\n" !n !bad;

    (* AS6: the working set, measured against the arithmetic. *)
    let tape = 2 * Deep_f32.t_max * Deep_f32.tw * Deep_f32.hs in
    let out0 = Deep_f32.t_max * Deep_f32.hs in
    let ws = (5 * Deep_f32.n_params) + tape + out0 in
    Printf.printf "   AS6  working set: 5 x %d params + %d tape + %d layer-0 outputs\n"
      Deep_f32.n_params tape out0;
    Printf.printf "        = %d floats = %.4f MiB at F32, %.4f MiB at F64\n"
      ws (float_of_int (ws * 4) /. 1048576.0) (float_of_int (ws * 8) /. 1048576.0);

    (* AS5: the budget, at two training-set sizes. *)
    seed_model ();
    Deep_f32.run_cycle 4 10 1;
    let s1 = Deep_f32.steps_taken.(0) in
    seed_model ();
    Deep_f32.run_cycle 4 120 1;
    let s2 = Deep_f32.steps_taken.(0) in
    seed_model ();
    Deep_f32.run_cycle 4 120 0;
    let s3 = Deep_f32.steps_taken.(0) in
    Printf.printf "   AS5  steps at T=10 %d, at T=120 %d (budget 4): %s; window refused -> %d\n"
      s1 s2 (if s1 = 4 && s2 = 4 then "EXACT" else "DIFFER") s3;

    (* AS4: the gradients under 51's criterion, S measured at F32. *)
    seed_model ();
    Deep_f32.forward_deep t;
    Deep_f32.backward_deep t;
    let s_mag = Deep_f32.s_out.(0) in
    let atol = k_const *. macheps32 *. s_mag /. h_rel in
    let stride = if Array.length Sys.argv > 3 then int_of_string Sys.argv.(3) else 1021 in
    let out = ref 0 and checked = ref 0 and worst = ref 0.0 and worst_ratio = ref 0.0 in
    let ga = Array.make Deep_f32.n_params 0.0 in
    for i = 0 to Deep_f32.n_params - 1 do
      ga.(i) <- Bigarray.Array1.unsafe_get Deep_f32.g i done;
    let i = ref 0 in
    while !i < Deep_f32.n_params do
      let pv = Bigarray.Array1.unsafe_get Deep_f32.p !i in
      let h = step_of pv in
      let fd = central_fd !i h t in
      let err = Float.abs (fd -. ga.(!i)) in
      let allowed = atol +. (rtol *. Float.abs ga.(!i)) in
      if err > allowed then incr out;
      if err > !worst then worst := err;
      if err /. allowed > !worst_ratio then worst_ratio := err /. allowed;
      incr checked; i := !i + stride
    done;
    Printf.printf "   AS4  T = %d, S = %.6e, atol = %.6e, h_rel = %.6e\n"
      t s_mag atol h_rel;
    Printf.printf "        outside %d of %d checked; worst |err| %.6e; worst err/allowed %.4f\n"
      !out !checked !worst !worst_ratio
  end

(* AS7b: a deliberately wrong gradient must be caught under the criterion AS4 used. *)
let () =
  if (if Array.length Sys.argv > 1 then Sys.argv.(1) else "all") = "all" then begin
    let t = if Array.length Sys.argv > 2 then int_of_string Sys.argv.(2) else 250 in
    seed_model ();
    Deep_f32.forward_deep t; Deep_f32.backward_deep t;
    let atol = k_const *. macheps32 *. Deep_f32.s_out.(0) /. h_rel in
    let idx = Deep_f32.o_hw in
    let good = Bigarray.Array1.unsafe_get Deep_f32.g idx in
    let bad = good +. (1e3 *. (atol +. (rtol *. Float.abs good)) +. 1e-6) in
    let h = step_of (Bigarray.Array1.unsafe_get Deep_f32.p idx) in
    let fd = central_fd idx h t in
    let caught = Float.abs (fd -. bad) > atol +. (rtol *. Float.abs bad) in
    Printf.printf "   AS7b a deliberately wrong head_w[0] gradient is caught: %s\n"
      (if caught then "YES" else "NO")
  end
