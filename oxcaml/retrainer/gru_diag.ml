(* Section 48 diagnostic: is G3's 1.3e-6 a wrong gradient or finite-difference noise?
 * It does NOT move G3's band -- 48.4 fixed that before the run and NO VERDICT stands.
 * It characterises the instrument, the way 47.13.3's X6 and 47.13.4's exception probe did.
 *
 * A WRONG gradient has an error floor that does not move with eps. FD NOISE is U-shaped:
 * truncation O(eps^2) dominates at large eps, roundoff O(macheps/eps) at small. *)
let hs = Gru_cell.h_size
let ins = Gru_cell.in_size
let g = Gru_cell.gw

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
  for i = 0 to hs - 1 do acc := !acc +. (coeff.(i) *. Gru_cell.h_new.(i)) done;
  !acc

let () =
  fill Gru_cell.w_ih 0.3 1; fill Gru_cell.w_hh 0.3 2;
  fill Gru_cell.b_ih 0.2 3; fill Gru_cell.b_hh 0.2 4;
  fill Gru_cell.x 1.0 5;    fill Gru_cell.h 1.0 6;
  fill coeff 1.0 7;
  Gru_cell.forward ();
  Array.blit coeff 0 Gru_cell.g_h_new 0 hs;
  Gru_cell.zero_grads ();
  Gru_cell.backward ();
  let a = Array.copy Gru_cell.g_w_ih in

  Printf.printf "== Section 48 diagnostic: eps sweep on w_ih (%d entries) ==\n" (g * ins);
  Printf.printf "   %-10s %-12s %-12s %-12s\n" "eps" "worst rel" "worst abs" "worst rel |g|>1e-3";
  List.iter (fun eps ->
    let wr = ref 0.0 and wa = ref 0.0 and wrb = ref 0.0 in
    for i = 0 to (g * ins) - 1 do
      let saved = Gru_cell.w_ih.(i) in
      Gru_cell.w_ih.(i) <- saved +. eps; let lp = loss () in
      Gru_cell.w_ih.(i) <- saved -. eps; let lm = loss () in
      Gru_cell.w_ih.(i) <- saved;
      let fd = (lp -. lm) /. (2.0 *. eps) in
      let an = a.(i) in
      let ab = Float.abs (fd -. an) in
      let den = Float.max (Float.abs fd) (Float.max (Float.abs an) 1e-9) in
      let rel = ab /. den in
      if rel > !wr then wr := rel;
      if ab > !wa then wa := ab;
      if Float.abs an > 1e-3 && rel > !wrb then wrb := rel
    done;
    Printf.printf "   %-10.0e %-12.3e %-12.3e %-12.3e\n" eps !wr !wa !wrb)
    [1e-3; 1e-4; 1e-5; 1e-6; 1e-7];
  Printf.printf "\n   A WRONG gradient shows a floor that does not move with eps.\n";
  Printf.printf "   FD NOISE is U-shaped: truncation at large eps, roundoff at small.\n"
