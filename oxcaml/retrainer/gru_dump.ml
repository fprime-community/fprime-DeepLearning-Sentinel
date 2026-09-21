(* Section 49, Arm B: dump the cell's weights, inputs and forward output at full precision
 * so that src/sentinel_models/reference.py's own gru_cell can be run on the same numbers
 * in float64 (49.3).
 *
 * (!) THIS LIVES OUTSIDE Gru_cell's ANNOTATED FUNCTIONS and does not edit them. J6 re-runs
 * the [@zero_alloc strict] build to prove 48's result is undisturbed by the dump path. *)

let hs = Gru_cell.h_size
let ins = Gru_cell.in_size
let g = Gru_cell.gw

(* 48's fill, unchanged: the same cell on the same numbers. *)
let fill arr scale off =
  let s = ref (12345 + off) in
  for i = 0 to Array.length arr - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    arr.(i) <- scale *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0)
  done

let emit oc name arr =
  Printf.fprintf oc "%s %d\n" name (Array.length arr);
  Array.iter (fun v -> Printf.fprintf oc "%.17g\n" v) arr

let () =
  let path = if Array.length Sys.argv > 1 then Sys.argv.(1) else "s49_cell.txt" in
  fill Gru_cell.w_ih 0.3 1;
  fill Gru_cell.w_hh 0.3 2;
  fill Gru_cell.b_ih 0.2 3;
  fill Gru_cell.b_hh 0.2 4;
  fill Gru_cell.x 1.0 5;
  fill Gru_cell.h 1.0 6;
  Gru_cell.forward ();
  let oc = open_out path in
  Printf.fprintf oc "dims %d %d %d\n" hs ins g;
  emit oc "w_ih" Gru_cell.w_ih;
  emit oc "w_hh" Gru_cell.w_hh;
  emit oc "b_ih" Gru_cell.b_ih;
  emit oc "b_hh" Gru_cell.b_hh;
  emit oc "x" Gru_cell.x;
  emit oc "h" Gru_cell.h;
  emit oc "h_new" Gru_cell.h_new;
  close_out oc;
  Printf.printf "   dumped hidden %d, inputs %d, gate width %d -> %s\n" hs ins g path
