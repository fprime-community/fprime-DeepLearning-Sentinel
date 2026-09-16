(* Section 49, Arm A: the gradient check with its step rule, step and criterion DECLARED
 * BEFORE THE RUN (49.2). This module is not annotated and is not part of 48's claim; only
 * Gru_cell's three functions are, and this file does not edit them.
 *
 * (!) STOP 15: nothing here is adjusted after a number is seen. The four constants below
 * are 49.2's, transcribed, and if they turn out to be wrong the run is reported as a loser
 * and they are re-declared in a successor section rather than edited here.
 *
 * The loss is L = sum_i coeff_i * h'_i, so dL/dh'_i = coeff_i is exactly what backward()
 * is handed -- the same loss 48 used, on the same fill, so this is the same cell. *)

let g = Gru_cell.gw
let hs = Gru_cell.h_size
let ins = Gru_cell.in_size

(* 48's fill, unchanged, so that Arm A measures 48's cell on 48's numbers. *)
let fill arr scale off =
  let s = ref (12345 + off) in
  for i = 0 to Array.length arr - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    arr.(i) <- scale *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0)
  done

let coeff = Array.make hs 0.0

let loss () =
  Gru_cell.forward ();
  let acc = ref 0.0 in
  for i = 0 to hs - 1 do
    acc := !acc +. (coeff.(i) *. Gru_cell.h_new.(i))
  done;
  !acc

(* -- 49.2's declared constants ------------------------------------------------------- *)
let macheps = epsilon_float                      (* 2^-52 = 2.220446049250313e-16 *)
let h_rel_declared = Float.cbrt macheps          (* 6.055454e-06, NOT 1e-4 *)
let rtol = 1e-6                                  (* carried unchanged from 48.4 *)
let k_const = 4.0                                (* small declared constant, not fitted *)

(* step rule: relative, per-parameter, as 48.2 registered and gru_check.ml did not do *)
let step_of p = h_rel_declared *. Float.max (Float.abs p) 1.0

(* -- accumulators --------------------------------------------------------------------- *)
let atol = ref 0.0
let outside = ref 0
let worst_margin = ref neg_infinity
let worst_margin_name = ref "-"
let items = ref []                               (* itemised failures, 49.6 *)
let j2_worst_rel = ref 0.0                       (* 48.4's criterion, at the new step *)
let j2_worst_name = ref "-"
let worst_abs = ref 0.0
let checked = ref 0

let central_fd arr idx h =
  let saved = arr.(idx) in
  arr.(idx) <- saved +. h;
  let lp = loss () in
  arr.(idx) <- saved -. h;
  let lm = loss () in
  arr.(idx) <- saved;
  (lp -. lm) /. (2.0 *. h)

let check_one name arr idx analytic =
  let h = step_of arr.(idx) in
  let fd = central_fd arr idx h in
  let err = Float.abs (fd -. analytic) in
  let allowed = !atol +. (rtol *. Float.abs analytic) in
  let margin = err -. allowed in
  incr checked;
  if err > !worst_abs then worst_abs := err;
  if margin > 0.0 then begin
    incr outside;
    if List.length !items < 24 then
      items := (name, idx, analytic, fd, err, allowed) :: !items
  end;
  if margin > !worst_margin then begin
    worst_margin := margin; worst_margin_name := name
  end;
  (* J2: 48.4's own criterion, unchanged, evaluated at the new step *)
  let denom = Float.max (Float.abs fd) (Float.max (Float.abs analytic) 1e-9) in
  let rel = err /. denom in
  if rel > !j2_worst_rel then begin j2_worst_rel := rel; j2_worst_name := name end

(* J3: the sweep, worst ABSOLUTE error at each relative step *)
let sweep_worst_abs h_rel analytics =
  let w = ref 0.0 in
  let one arr idx analytic =
    let h = h_rel *. Float.max (Float.abs arr.(idx)) 1.0 in
    let fd = central_fd arr idx h in
    let e = Float.abs (fd -. analytic) in
    if e > !w then w := e in
  let (a_w_ih, a_w_hh, a_b_ih, a_b_hh, a_x, a_h) = analytics in
  for i = 0 to (g * ins) - 1 do one Gru_cell.w_ih i a_w_ih.(i) done;
  for i = 0 to (g * hs) - 1 do one Gru_cell.w_hh i a_w_hh.(i) done;
  for i = 0 to g - 1 do one Gru_cell.b_ih i a_b_ih.(i) done;
  for i = 0 to g - 1 do one Gru_cell.b_hh i a_b_hh.(i) done;
  for i = 0 to ins - 1 do one Gru_cell.x i a_x.(i) done;
  for i = 0 to hs - 1 do one Gru_cell.h i a_h.(i) done;
  !w

let () =
  fill Gru_cell.w_ih 0.3 1;
  fill Gru_cell.w_hh 0.3 2;
  fill Gru_cell.b_ih 0.2 3;
  fill Gru_cell.b_hh 0.2 4;
  fill Gru_cell.x 1.0 5;
  fill Gru_cell.h 1.0 6;
  fill coeff 1.0 7;

  (* the analytic gradient, once *)
  Gru_cell.forward ();
  Array.blit coeff 0 Gru_cell.g_h_new 0 hs;
  Gru_cell.zero_grads ();
  Gru_cell.backward ();
  let a_w_ih = Array.copy Gru_cell.g_w_ih in
  let a_w_hh = Array.copy Gru_cell.g_w_hh in
  let a_b_ih = Array.copy Gru_cell.g_b_ih in
  let a_b_hh = Array.copy Gru_cell.g_b_hh in
  let a_x = Array.copy Gru_cell.g_x in
  let a_h = Array.copy Gru_cell.g_h in

  (* L_scale is measured at the initial parameters, BEFORE any gradient is compared *)
  let l_scale = Float.abs (loss ()) in
  atol := k_const *. macheps *. l_scale /. h_rel_declared;

  Printf.printf "== Section 49, Arm A: the check specified before it ran ==\n";
  Printf.printf "   hidden %d, inputs %d, gate width %d\n" hs ins g;
  Printf.printf "   DECLARED IN 49.2, BEFORE THE RUN:\n";
  Printf.printf "     macheps            %.17g\n" macheps;
  Printf.printf "     h_rel = cbrt       %.6e   (step h(p) = h_rel * max(|p|,1))\n"
    h_rel_declared;
  Printf.printf "     rtol               %.1e   (unchanged from 48.4)\n" rtol;
  Printf.printf "     K                  %.1f\n" k_const;
  Printf.printf "   MEASURED AT INIT, BEFORE ANY GRADIENT IS COMPARED:\n";
  Printf.printf "     L_scale            %.6e\n" l_scale;
  Printf.printf "     atol = K*eps*L/h   %.6e\n" !atol;

  for i = 0 to (g * ins) - 1 do check_one "w_ih" Gru_cell.w_ih i a_w_ih.(i) done;
  for i = 0 to (g * hs) - 1 do check_one "w_hh" Gru_cell.w_hh i a_w_hh.(i) done;
  for i = 0 to g - 1 do check_one "b_ih" Gru_cell.b_ih i a_b_ih.(i) done;
  for i = 0 to g - 1 do check_one "b_hh" Gru_cell.b_hh i a_b_hh.(i) done;
  for i = 0 to ins - 1 do check_one "x" Gru_cell.x i a_x.(i) done;
  for i = 0 to hs - 1 do check_one "h" Gru_cell.h i a_h.(i) done;

  Printf.printf "\n-- J1: abs(fd-g) <= atol + rtol*abs(g), over all %d\n" !checked;
  Printf.printf "   entries outside   %d\n" !outside;
  Printf.printf "   worst abs error   %.6e\n" !worst_abs;
  Printf.printf "   worst margin      %.6e  (on %s; negative means inside)\n"
    !worst_margin !worst_margin_name;
  if !outside > 0 then begin
    Printf.printf "   ITEMISED (49.6: a count that is not enumerable is not a count):\n";
    List.iter (fun (nm, i, a, fd, e, al) ->
      Printf.printf "     %-5s [%6d]  analytic %+.9e  fd %+.9e  err %.3e  allowed %.3e\n"
        nm i a fd e al) (List.rev !items);
    if !outside > List.length !items then
      Printf.printf "     ... and %d more not listed\n" (!outside - List.length !items)
  end;
  let j1 =
    if !outside = 0 then "HOLD"
    else if !outside <= 10 then "NO VERDICT"
    else "FAIL" in
  Printf.printf "   band: 0 outside HOLD, <=10 NO VERDICT, else FAIL  -> %s\n" j1;

  Printf.printf "\n-- J2: 48.4's OWN criterion, at the new step, reported whatever it says\n";
  Printf.printf "   worst relative error %.6e  (on %s)\n" !j2_worst_rel !j2_worst_name;
  Printf.printf "   48.8 recorded 1.309e-06 at its own absolute step of 1e-5.\n";
  Printf.printf "   48's G3 stands at NO VERDICT. This number does not replace it.\n";

  Printf.printf "\n-- J3: the sweep, worst ABSOLUTE error against the relative step\n";
  let analytics = (a_w_ih, a_w_hh, a_b_ih, a_b_hh, a_x, a_h) in
  let steps = [| 1e-3; 1e-4; 1e-5; 1e-6; 1e-7 |] in
  let vals = Array.map (fun s -> sweep_worst_abs s analytics) steps in
  Array.iteri (fun i s -> Printf.printf "     h_rel %.0e   worst abs %.6e\n" s vals.(i))
    steps;
  let mi = ref 0 in
  Array.iteri (fun i v -> if v < vals.(!mi) then mi := i) vals;
  Printf.printf "   minimum at h_rel %.0e\n" steps.(!mi);
  let j3 =
    if !mi = 0 || !mi = Array.length steps - 1 then "NO VERDICT (minimum at an endpoint)"
    else "HOLD (interior minimum: a U, not a floor)" in
  Printf.printf "   -> %s\n" j3;

  exit (if !outside = 0 then 0 else if !outside <= 10 then 1 else 2)
