(* Section 54: FT3's dump and FT6's comparison against 48's float64 cell. *)

let fill_f32 (a : Gru_f32.f32) scale off =
  let s = ref (12345 + off) in
  for i = 0 to Bigarray.Array1.dim a - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    (* the C++ side computes in double then rounds to F32; so does this *)
    Bigarray.Array1.set a i (scale *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0))
  done

let fill64 arr scale off =
  let s = ref (12345 + off) in
  for i = 0 to Array.length arr - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    arr.(i) <- scale *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0)
  done

let () =
  fill_f32 Gru_f32.w_ih 0.3 1; fill_f32 Gru_f32.w_hh 0.3 2;
  fill_f32 Gru_f32.b_ih 0.2 3; fill_f32 Gru_f32.b_hh 0.2 4;
  fill_f32 Gru_f32.x 1.0 5;    fill_f32 Gru_f32.h 1.0 6;
  Gru_f32.forward ();
  (* the backward must RUN, not merely compile: FT2 is about strict, but a backward that
     cannot be executed would make the verdict worthless *)
  for i = 0 to Gru_f32.hs - 1 do
    Bigarray.Array1.set Gru_f32.g_h_new i 1.0
  done;
  Gru_f32.zero_grads ();
  Gru_f32.backward ();
  let oc = open_out Sys.argv.(1) in
  for i = 0 to Gru_f32.hs - 1 do
    Printf.fprintf oc "%.17g\n" (Bigarray.Array1.get Gru_f32.h_new i)
  done;
  close_out oc;

  (* FT6: the same algebra at float64, 48's cell *)
  fill64 Gru_cell.w_ih 0.3 1; fill64 Gru_cell.w_hh 0.3 2;
  fill64 Gru_cell.b_ih 0.2 3; fill64 Gru_cell.b_hh 0.2 4;
  fill64 Gru_cell.x 1.0 5;    fill64 Gru_cell.h 1.0 6;
  Gru_cell.forward ();
  let worst = ref 0.0 and idx = ref 0 in
  for i = 0 to Gru_f32.hs - 1 do
    let d = Float.abs ((Bigarray.Array1.get Gru_f32.h_new i) -. Gru_cell.h_new.(i)) in
    if d > !worst then begin worst := d; idx := i end
  done;
  Printf.printf "== Section 54, FT6: the F32 cell against 48's F64 cell ==\n";
  Printf.printf "   worst |F32 - F64| = %.6e at index %d\n" !worst !idx;
  let v = if !worst <= 1e-6 then "HOLD" else if !worst <= 1e-4 then "NO VERDICT" else "FAIL" in
  Printf.printf "   band: <=1e-6 HOLD, <=1e-4 NO VERDICT, else FAIL -> %s\n" v;
  Printf.printf "   (the two differ in accumulation order as well as precision: 48 starts the\n";
  Printf.printf "    accumulator at the bias, flight/ and this cell start at zero)\n"
