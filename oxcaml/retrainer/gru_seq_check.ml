(* Section 50, U3 and U6: the BPTT gradients, under 49's criterion reused VERBATIM.
 *
 * (!) STOP 15: h_rel, rtol and K are 49.2's and are not adjusted after a number is seen.
 * Only L_scale is re-measured, because the loss is now a sum over t_steps (50.3). *)

let hs = Gru_cell.h_size
let ins = Gru_cell.in_size
let g = Gru_cell.gw

(* 48's fill, unchanged. Because it is a sequential recurrence from the seed, the first 16
   values of x_seq are bit-identical to the 16 values 48 and 49 gave Gru_cell.x, and the
   first 80 of coeff_seq to their coeff. That is what makes U6 a real comparison. *)
let fill arr scale off =
  let s = ref (12345 + off) in
  for i = 0 to Array.length arr - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    arr.(i) <- scale *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0)
  done

(* -- 49.2's declared constants, verbatim ----------------------------------------------- *)
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
let worst_margin_name = ref "-"
let items = ref []
let checked = ref 0

let loss () = Gru_seq.forward_seq !t_steps; Gru_seq.loss_out.(0)

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
  end

let () =
  t_steps := (if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 250);
  let t = !t_steps in
  fill Gru_cell.w_ih 0.3 1;
  fill Gru_cell.w_hh 0.3 2;
  fill Gru_cell.b_ih 0.2 3;
  fill Gru_cell.b_hh 0.2 4;
  fill Gru_seq.x_seq 1.0 5;
  fill Gru_seq.h0 1.0 6;
  fill Gru_seq.coeff_seq 1.0 7;

  (* the analytic gradient, once *)
  Gru_seq.forward_seq t;
  Gru_seq.backward_seq t;
  let a_w_ih = Array.copy Gru_cell.g_w_ih in
  let a_w_hh = Array.copy Gru_cell.g_w_hh in
  let a_b_ih = Array.copy Gru_cell.g_b_ih in
  let a_b_hh = Array.copy Gru_cell.g_b_hh in
  let a_x = Array.copy Gru_seq.g_x_seq in
  let a_h0 = Array.copy Gru_seq.g_h0 in

  let l_scale = Float.abs (loss ()) in
  atol := k_const *. macheps *. l_scale /. h_rel_declared;

  Printf.printf "== Section 50: BPTT over a fixed tape, T = %d ==\n" t;
  Printf.printf "   hidden %d, inputs %d, gate width %d, tape %d floats\n"
    hs ins g (Array.length Gru_seq.tape);
  Printf.printf "   49.2's constants, reused verbatim: h_rel %.6e, rtol %.1e, K %.1f\n"
    h_rel_declared rtol k_const;
  Printf.printf "   L_scale re-measured for this loss: %.6e   atol %.6e\n" l_scale !atol;

  for i = 0 to (g * ins) - 1 do check_one "w_ih" Gru_cell.w_ih i a_w_ih.(i) done;
  for i = 0 to (g * hs) - 1 do check_one "w_hh" Gru_cell.w_hh i a_w_hh.(i) done;
  for i = 0 to g - 1 do check_one "b_ih" Gru_cell.b_ih i a_b_ih.(i) done;
  for i = 0 to g - 1 do check_one "b_hh" Gru_cell.b_hh i a_b_hh.(i) done;
  for i = 0 to (t * ins) - 1 do check_one "x_seq" Gru_seq.x_seq i a_x.(i) done;
  for i = 0 to hs - 1 do check_one "h0" Gru_seq.h0 i a_h0.(i) done;

  Printf.printf "\n-- U3: abs(fd-g) <= atol + rtol*abs(g), over all %d\n" !checked;
  Printf.printf "   entries outside   %d\n" !outside;
  Printf.printf "   worst abs error   %.6e\n" !worst_abs;
  Printf.printf "   worst margin      %.6e  (on %s; negative means inside)\n"
    !worst_margin !worst_margin_name;
  if !outside > 0 then begin
    Printf.printf "   ITEMISED:\n";
    List.iter (fun (nm, i, a, fd, e, al) ->
      Printf.printf "     %-6s [%6d]  analytic %+.9e  fd %+.9e  err %.3e  allowed %.3e\n"
        nm i a fd e al) (List.rev !items);
    if !outside > List.length !items then
      Printf.printf "     ... and %d more not listed\n" (!outside - List.length !items)
  end;
  let verdict =
    if !outside = 0 then "HOLD"
    else if !outside <= 10 then "NO VERDICT"
    else "FAIL" in
  Printf.printf "   band: 0 outside HOLD, <=10 NO VERDICT, else FAIL  -> %s\n" verdict;
  exit (if !outside = 0 then 0 else if !outside <= 10 then 1 else 2)
