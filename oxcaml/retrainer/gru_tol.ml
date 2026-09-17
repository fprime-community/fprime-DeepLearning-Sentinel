(* Section 51: the tolerance model, derived at 51.1 rather than fitted.
 *
 * (!) EXACTLY ONE QUANTITY CHANGES FROM 49.2 (51.2). h_rel, rtol and K are carried verbatim.
 * The scale becomes S = sum_t sum_i |c_{t,i} * h_{t,i}| -- the ACCUMULATED MAGNITUDE -- where
 * 49 and 50 used |L|, the signed total. The classical summation bound carries sum |a_i|
 * because cancellation does not cancel the errors.
 *
 * (!) STOP 15: nothing here is adjusted after a number is seen.
 * gru_cell.ml and gru_seq.ml are NOT edited; this module only measures them. *)

let hs = Gru_cell.h_size
let ins = Gru_cell.in_size
let g = Gru_cell.gw

let fill arr scale off =
  let s = ref (12345 + off) in
  for i = 0 to Array.length arr - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    arr.(i) <- scale *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0)
  done

(* -- 49.2's constants, carried verbatim ------------------------------------------------ *)
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

(* S = sum_t sum_i |c * h_t|. h_t is tape h_in[t+1] for t < T-1, and h_new for t = T-1.
   Must be called with the tape holding a completed forward pass. *)
let accumulated_magnitude t =
  let s = ref 0.0 in
  for tt = 0 to t - 1 do
    for i = 0 to hs - 1 do
      let h_t =
        if tt = t - 1 then Gru_cell.h_new.(i)
        else Gru_seq.tape.(Gru_seq.tix (tt + 1) 4 i) in
      s := !s +. Float.abs (Gru_seq.coeff_seq.((tt * hs) + i) *. h_t)
    done
  done;
  !s

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
    if List.length !items < 16 then
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

  Gru_seq.forward_seq t;
  Gru_seq.backward_seq t;
  let a_w_ih = Array.copy Gru_cell.g_w_ih in
  let a_w_hh = Array.copy Gru_cell.g_w_hh in
  let a_b_ih = Array.copy Gru_cell.g_b_ih in
  let a_b_hh = Array.copy Gru_cell.g_b_hh in
  let a_x = Array.copy Gru_seq.g_x_seq in
  let a_h0 = Array.copy Gru_seq.g_h0 in

  let l_signed = Float.abs (loss ()) in            (* what 49 and 50 used *)
  let s_mag = accumulated_magnitude t in           (* what 51.1 derives *)
  atol := k_const *. macheps *. s_mag /. h_rel_declared;

  Printf.printf "== Section 51: the derived tolerance, T = %d ==\n" t;
  Printf.printf "   S (accumulated magnitude) %.6e   abs(L) (signed total) %.6e   ratio %.1f\n"
    s_mag l_signed (s_mag /. l_signed);
  Printf.printf "   atol %.6e   (49/50 would have used %.6e)\n"
    !atol (k_const *. macheps *. l_signed /. h_rel_declared);

  for i = 0 to (g * ins) - 1 do check_one "w_ih" Gru_cell.w_ih i a_w_ih.(i) done;
  for i = 0 to (g * hs) - 1 do check_one "w_hh" Gru_cell.w_hh i a_w_hh.(i) done;
  for i = 0 to g - 1 do check_one "b_ih" Gru_cell.b_ih i a_b_ih.(i) done;
  for i = 0 to g - 1 do check_one "b_hh" Gru_cell.b_hh i a_b_hh.(i) done;
  for i = 0 to (t * ins) - 1 do check_one "x_seq" Gru_seq.x_seq i a_x.(i) done;
  for i = 0 to hs - 1 do check_one "h0" Gru_seq.h0 i a_h0.(i) done;

  Printf.printf "   checked %d   entries outside %d\n" !checked !outside;
  Printf.printf "   worst abs error %.6e   worst margin %.6e (%s)\n"
    !worst_abs !worst_margin !worst_margin_name;
  if !outside > 0 then begin
    Printf.printf "   ITEMISED:\n";
    List.iter (fun (nm, i, a, fd, e, al) ->
      Printf.printf "     %-6s [%6d]  analytic %+.9e  fd %+.9e  err %.3e  allowed %.3e\n"
        nm i a fd e al) (List.rev !items);
    if !outside > List.length !items then
      Printf.printf "     ... and %d more not listed\n" (!outside - List.length !items)
  end;
  exit (if !outside = 0 then 0 else if !outside <= 10 then 1 else 2)
