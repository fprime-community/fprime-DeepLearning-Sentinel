(* Section 57's checker: HL1 to HL6. Not annotated and not part of the claim.
 *
 * (!) STOP 26: h_rel, rtol and K are 49.2's and 51.2's, carried verbatim from
 * gru_deep_check.ml. Only S changes, which is the whole point of the section.
 * (!) STOP 27: coeff_h is ZERO throughout. This is the head-only loss. *)

let hs = Head57.hs
let ins = Head57.ins

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
let loss () = Head57.forward_deep !t_steps; Head57.loss_out.(0)

let central_fd arr idx h =
  let saved = arr.(idx) in
  arr.(idx) <- saved +. h; let lp = loss () in
  arr.(idx) <- saved -. h; let lm = loss () in
  arr.(idx) <- saved;
  (lp -. lm) /. (2.0 *. h)

let seed_model () =
  fill Head57.w_ih0 0.1 1; fill Head57.w_hh0 0.1 2;
  fill Head57.b_ih0 0.1 3; fill Head57.b_hh0 0.1 4;
  fill Head57.w_ih1 0.1 11; fill Head57.w_hh1 0.1 12;
  fill Head57.b_ih1 0.1 13; fill Head57.b_hh1 0.1 14;
  fill Head57.head_w 0.1 21; fill Head57.head_b 0.1 22;
  fill Head57.h_init0 0.1 5; fill Head57.h_init1 0.1 6;
  fill Head57.x_seq 1.0 8;
  (* (!) HEAD-ONLY: every coeff_h is zero. Stop 27. *)
  Array.fill Head57.coeff_h 0 (Array.length Head57.coeff_h) 0.0;
  fill Head57.coeff_y 1.0 17

(* Run the analytic pass and return the three S values. *)
let measure t =
  t_steps := t;
  seed_model ();
  Head57.forward_deep t;
  Head57.backward_deep t;
  (Head57.s_out.(0), Head57.s_flat.(0), Head57.s_rec.(0))

let blocks () = [
  ("w_ih0", Head57.w_ih0, Array.copy Head57.g_w_ih0);
  ("w_hh0", Head57.w_hh0, Array.copy Head57.g_w_hh0);
  ("b_ih0", Head57.b_ih0, Array.copy Head57.g_b_ih0);
  ("b_hh0", Head57.b_hh0, Array.copy Head57.g_b_hh0);
  ("w_ih1", Head57.w_ih1, Array.copy Head57.g_w_ih1);
  ("w_hh1", Head57.w_hh1, Array.copy Head57.g_w_hh1);
  ("b_ih1", Head57.b_ih1, Array.copy Head57.g_b_ih1);
  ("b_hh1", Head57.b_hh1, Array.copy Head57.g_b_hh1);
  ("head_w", Head57.head_w, Array.copy Head57.g_head_w);
  ("head_b", Head57.head_b, Array.copy Head57.g_head_b);
]

(* Count entries outside 49's criterion under a given S. `stride` subsamples the
   large blocks; it is declared here and identical for every model compared. *)
let count_outside s_mag stride =
  let atol = k_const *. macheps *. s_mag /. h_rel_declared in
  List.map (fun (name, arr, g) ->
      let out = ref 0 and n = ref 0 in
      let i = ref 0 in
      while !i < Array.length arr do
        let h = step_of arr.(!i) in
        let fd = central_fd arr !i h in
        let err = Float.abs (fd -. g.(!i)) in
        if err > atol +. (rtol *. Float.abs g.(!i)) then incr out;
        incr n; i := !i + stride
      done;
      (name, !out, !n))
    (blocks ())

let () =
  let stride = if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 97 in

  Printf.printf "   HL3/HL5  the three S at four T (head-only, coeff_h = 0)\n";
  Printf.printf "     %5s %14s %14s %14s %10s %10s\n"
    "T" "M-51" "M-flat" "M-rec" "flat/51" "rec/flat";
  let ratios = ref [] and flat_gt = ref true in
  List.iter (fun t ->
      let (s51, sflat, srec) = measure t in
      if not (sflat > s51) then flat_gt := false;
      ratios := (t, srec /. sflat) :: !ratios;
      Printf.printf "     %5d %14.6e %14.6e %14.6e %10.3f %10.3f\n"
        t s51 sflat srec (sflat /. s51) (srec /. sflat))
    [1; 10; 50; 250];
  Printf.printf "   HL3  M-flat > M-51 at every T: %s\n" (if !flat_gt then "YES" else "NO");
  let r = List.rev !ratios in
  let mono = ref true in
  let rec walk = function
    | (_, a) :: ((_, b) :: _ as rest) -> if not (b > a) then mono := false; walk rest
    | _ -> () in
  walk r;
  Printf.printf "   HL5  rec/flat increases monotonically with T: %s\n"
    (if !mono then "YES" else "NO");

  (* HL2 and HL4 at the flown T. *)
  Printf.printf "\n   HL2/HL4  gradient entries outside 49's criterion at T = 250 (stride %d)\n"
    stride;
  let (s51, sflat, srec) = measure 250 in
  let run label s =
    let rows = count_outside s stride in
    let tot = List.fold_left (fun a (_, o, _) -> a + o) 0 rows in
    let n = List.fold_left (fun a (_, _, c) -> a + c) 0 rows in
    Printf.printf "     %-8s atol %.6e   outside %d of %d\n" label
      (k_const *. macheps *. s /. h_rel_declared) tot n;
    List.iter (fun (nm, o, c) ->
        if o > 0 then Printf.printf "        %-8s %d of %d\n" nm o c) rows;
    tot in
  let o51 = run "M-51" s51 in
  let oflat = run "M-flat" sflat in
  let orec = run "M-rec" srec in
  Printf.printf "   HL2  M-51 under-covers (>= 1 outside): %s   [%d]\n"
    (if o51 >= 1 then "YES" else "NO") o51;
  Printf.printf "   HL4  M-rec covers all entries: %s   [M-flat left %d, M-rec left %d]\n"
    (if orec = 0 then "YES" else "NO") oflat orec;

  (* HL6: a deliberately wrong gradient must be caught under M-rec. *)
  let g = Array.copy Head57.g_head_w in
  let idx = 0 in
  let atol = k_const *. macheps *. srec /. h_rel_declared in
  let bad = g.(idx) +. (1e3 *. (atol +. (rtol *. Float.abs g.(idx))) +. 1e-6) in
  let h = step_of Head57.head_w.(idx) in
  let fd = central_fd Head57.head_w idx h in
  let caught = Float.abs (fd -. bad) > atol +. (rtol *. Float.abs bad) in
  Printf.printf "\n   HL6  a deliberately wrong head_w[0] is caught under M-rec: %s\n"
    (if caught then "YES" else "NO")

(* Reported because "0 outside" does not say by how much. 57.6: the margin decides
   whether this result survives a change of precision, and E5-b is at F32 where the
   unit roundoff is about 5e8 times larger. *)
let () =
  let stride = if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 97 in
  let (s51, _, _) = measure 250 in
  let atol = k_const *. macheps *. s51 /. h_rel_declared in
  let worst_err = ref 0.0 and worst_ratio = ref 0.0 and worst_nm = ref "-" in
  List.iter (fun (name, arr, g) ->
      let i = ref 0 in
      while !i < Array.length arr do
        let h = step_of arr.(!i) in
        let fd = central_fd arr !i h in
        let err = Float.abs (fd -. g.(!i)) in
        let allowed = atol +. (rtol *. Float.abs g.(!i)) in
        if err > !worst_err then worst_err := err;
        if err /. allowed > !worst_ratio then begin
          worst_ratio := err /. allowed; worst_nm := name end;
        i := !i + stride
      done) (blocks ());
  Printf.printf "\n   headroom under M-51: worst |err| %.6e, worst err/allowed %.4f on %s\n"
    !worst_err !worst_ratio !worst_nm;
  Printf.printf "   the criterion's two terms: atol %.6e, and rtol|g| carries the rest\n" atol
