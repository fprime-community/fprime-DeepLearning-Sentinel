(* Section 48's correctness check: G3, central finite differences over ALL parameters.
 *
 * (!) A STRICT VERDICT ON WRONG ARITHMETIC IS WORTH NOTHING -- 48.5 says G1, G2 and G6 are
 * withdrawn if this fails. This module is not annotated and is not part of the claim; only
 * Gru_cell's functions are.
 *
 * The loss is L = sum_i coeff_i * h'_i with fixed coefficients, so dL/dh'_i = coeff_i is
 * exactly what backward() is handed. Central differences: (L(p+e) - L(p-e)) / 2e. *)

let g = Gru_cell.gw
let hs = Gru_cell.h_size
let ins = Gru_cell.in_size

(* A fixed, reproducible fill. No RNG dependency: a deterministic recurrence. *)
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

let worst_rel = ref 0.0
let worst_name = ref "-"
let checked = ref 0

let check_one name arr idx analytic =
  let eps = 1e-5 in
  let saved = arr.(idx) in
  arr.(idx) <- saved +. eps;
  let lp = loss () in
  arr.(idx) <- saved -. eps;
  let lm = loss () in
  arr.(idx) <- saved;
  let fd = (lp -. lm) /. (2.0 *. eps) in
  let denom = Float.max (Float.abs fd) (Float.max (Float.abs analytic) 1e-9) in
  let rel = Float.abs (fd -. analytic) /. denom in
  incr checked;
  if rel > !worst_rel then begin worst_rel := rel; worst_name := name end

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

  Printf.printf "== Section 48, G3: central finite differences ==\n";
  Printf.printf "   hidden %d, inputs %d, gate width %d\n" hs ins g;

  for i = 0 to (g * ins) - 1 do check_one "w_ih" Gru_cell.w_ih i a_w_ih.(i) done;
  for i = 0 to (g * hs) - 1 do check_one "w_hh" Gru_cell.w_hh i a_w_hh.(i) done;
  for i = 0 to g - 1 do check_one "b_ih" Gru_cell.b_ih i a_b_ih.(i) done;
  for i = 0 to g - 1 do check_one "b_hh" Gru_cell.b_hh i a_b_hh.(i) done;
  for i = 0 to ins - 1 do check_one "x" Gru_cell.x i a_x.(i) done;
  for i = 0 to hs - 1 do check_one "h" Gru_cell.h i a_h.(i) done;

  Printf.printf "   checked %d parameters and inputs\n" !checked;
  Printf.printf "   worst relative error %.3e  (on %s)\n" !worst_rel !worst_name;
  let verdict =
    if !worst_rel <= 1e-6 then "HOLD"
    else if !worst_rel <= 1e-4 then "NO VERDICT"
    else "FAIL" in
  Printf.printf "   band: <=1e-6 HOLD, <=1e-4 NO VERDICT, else FAIL  -> %s\n" verdict;
  (* the forward state, for G6 to compare against the flight core *)
  let oc = open_out "h_new.txt" in
  for i = 0 to hs - 1 do Printf.fprintf oc "%.17g\n" Gru_cell.h_new.(i) done;
  close_out oc;
  exit (if !worst_rel <= 1e-4 then 0 else 1)
