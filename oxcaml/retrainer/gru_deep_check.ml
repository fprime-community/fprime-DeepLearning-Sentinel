(* Section 52, HD4 and HD7: the two-layer + head gradients under 51's criterion carried
 * verbatim, and a dump for the reference.py comparison.
 *
 * (!) STOP 15: h_rel, rtol and K are 49.2's and 51.2's, unchanged. Only S is re-measured.
 * Parallelism is by disjoint slice of the parameter index space; each process re-fills
 * deterministically and computes the same analytic gradient, so the numbers are
 * BIT-IDENTICAL to a serial run (52.7). *)

let hs = Gru_deep.hs
let ins = Gru_deep.ins

let fill arr scale off =
  let s = ref (12345 + off) in
  for i = 0 to Array.length arr - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    arr.(i) <- scale *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0)
  done

let macheps = epsilon_float
let h_rel_declared = Float.cbrt macheps
let rtol = 1e-6
let k_const = 4.0
let step_of p = h_rel_declared *. Float.max (Float.abs p) 1.0

let t_steps = ref 0
let atol = ref 0.0
let outside = ref 0
let worst_abs = ref 0.0
let worst_margin = ref neg_infinity
let worst_name = ref "-"
let items = ref []
let checked = ref 0

let loss () = Gru_deep.forward_deep !t_steps; Gru_deep.loss_out.(0)

let central_fd arr idx h =
  let saved = arr.(idx) in
  arr.(idx) <- saved +. h; let lp = loss () in
  arr.(idx) <- saved -. h; let lm = loss () in
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
    if List.length !items < 12 then
      items := (name, idx, analytic, fd, err, allowed) :: !items
  end;
  if margin > !worst_margin then begin worst_margin := margin; worst_name := name end

let () =
  let slice = if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 0 in
  let nslice = if Array.length Sys.argv > 2 then int_of_string Sys.argv.(2) else 1 in
  t_steps := (if Array.length Sys.argv > 3 then int_of_string Sys.argv.(3) else 250);
  let dump = Array.length Sys.argv > 4 && Sys.argv.(4) = "dump" in
  let t = !t_steps in

  fill Gru_deep.w_ih0 0.3 1;  fill Gru_deep.w_hh0 0.3 2;
  fill Gru_deep.b_ih0 0.2 3;  fill Gru_deep.b_hh0 0.2 4;
  fill Gru_deep.w_ih1 0.2 11; fill Gru_deep.w_hh1 0.2 12;
  fill Gru_deep.b_ih1 0.1 13; fill Gru_deep.b_hh1 0.1 14;
  fill Gru_deep.head_w 0.1 21; fill Gru_deep.head_b 0.1 22;
  fill Gru_deep.x_seq 1.0 5;
  fill Gru_deep.h_init0 1.0 6; fill Gru_deep.h_init1 1.0 16;
  fill Gru_deep.coeff_h 1.0 7; fill Gru_deep.coeff_y 1.0 17;

  Gru_deep.forward_deep t;
  if dump then begin
    let oc = open_out Sys.argv.(5) in
    Printf.fprintf oc "dims %d %d %d %d %d\n" hs ins Gru_deep.gw Gru_deep.n_out t;
    let emit name arr =
      Printf.fprintf oc "%s %d\n" name (Array.length arr);
      Array.iter (fun v -> Printf.fprintf oc "%.17g\n" v) arr in
    emit "w_ih0" Gru_deep.w_ih0; emit "w_hh0" Gru_deep.w_hh0;
    emit "b_ih0" Gru_deep.b_ih0; emit "b_hh0" Gru_deep.b_hh0;
    emit "w_ih1" Gru_deep.w_ih1; emit "w_hh1" Gru_deep.w_hh1;
    emit "b_ih1" Gru_deep.b_ih1; emit "b_hh1" Gru_deep.b_hh1;
    emit "head_w" Gru_deep.head_w; emit "head_b" Gru_deep.head_b;
    emit "x_seq" (Array.sub Gru_deep.x_seq 0 (t * ins));
    emit "h_init0" Gru_deep.h_init0; emit "h_init1" Gru_deep.h_init1;
    emit "y" Gru_deep.y;
    close_out oc;
    Printf.printf "   dumped two layers and head, T = %d -> %s\n" t Sys.argv.(5);
    exit 0
  end;

  Gru_deep.backward_deep t;
  let segs = [|
    ("w_ih0", Gru_deep.w_ih0, Array.copy Gru_deep.g_w_ih0);
    ("w_hh0", Gru_deep.w_hh0, Array.copy Gru_deep.g_w_hh0);
    ("b_ih0", Gru_deep.b_ih0, Array.copy Gru_deep.g_b_ih0);
    ("b_hh0", Gru_deep.b_hh0, Array.copy Gru_deep.g_b_hh0);
    ("w_ih1", Gru_deep.w_ih1, Array.copy Gru_deep.g_w_ih1);
    ("w_hh1", Gru_deep.w_hh1, Array.copy Gru_deep.g_w_hh1);
    ("b_ih1", Gru_deep.b_ih1, Array.copy Gru_deep.g_b_ih1);
    ("b_hh1", Gru_deep.b_hh1, Array.copy Gru_deep.g_b_hh1);
    ("head_w", Gru_deep.head_w, Array.copy Gru_deep.g_head_w);
    ("head_b", Gru_deep.head_b, Array.copy Gru_deep.g_head_b);
    ("x_seq", Gru_deep.x_seq, Array.copy Gru_deep.g_x_seq);
    ("h_init0", Gru_deep.h_init0, Array.copy Gru_deep.g_h_init0);
    ("h_init1", Gru_deep.h_init1, Array.copy Gru_deep.g_h_init1);
  |] in
  let seg_len k = let (nm, a, _) = segs.(k) in
    if nm = "x_seq" then t * ins else Array.length a in
  let total = ref 0 in
  Array.iteri (fun k _ -> total := !total + seg_len k) segs;

  let s_mag = Gru_deep.s_out.(0) in
  atol := k_const *. macheps *. s_mag /. h_rel_declared;

  if slice = 0 then begin
    Printf.printf "== Section 52: two layers and the head, T = %d ==\n" t;
    Printf.printf "   parameters in the index space %d   S %.6e   atol %.6e\n"
      !total s_mag !atol;
    Printf.printf "   51.2's constants verbatim: h_rel %.6e rtol %.1e K %.1f\n"
      h_rel_declared rtol k_const
  end;

  let gi = ref 0 in
  Array.iteri (fun k (nm, arr, ana) ->
    let n = seg_len k in
    for i = 0 to n - 1 do
      if !gi mod nslice = slice then check_one nm arr i ana.(i);
      incr gi
    done) segs;

  Printf.printf "SLICE %d/%d checked %d outside %d worst_abs %.6e worst_margin %.6e (%s)\n"
    slice nslice !checked !outside !worst_abs !worst_margin !worst_name;
  List.iter (fun (nm, i, a, fd, e, al) ->
    Printf.printf "  OUT %-8s [%6d] analytic %+.9e fd %+.9e err %.3e allowed %.3e\n"
      nm i a fd e al) (List.rev !items);
  exit (if !outside = 0 then 0 else 2)
