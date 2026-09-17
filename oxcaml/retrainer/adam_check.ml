(* Section 53: AD3 (the trajectory) and AD6 (one full training step). *)

let n = Adam.n_max

let fill arr scale off =
  let s = ref (12345 + off) in
  for i = 0 to Array.length arr - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    arr.(i) <- scale *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0)
  done

let () =
  let mode = if Array.length Sys.argv > 1 then Sys.argv.(1) else "traj" in
  if mode = "traj" then begin
    (* AD3: 20 updates over 75,360 parameters on a fixed synthetic gradient sequence.
       The gradient at step t is the same LCG fill at offset 1000 + t, so the reference
       reproduces it without a 30 MB dump. *)
    let steps = 20 in
    let p = Array.make n 0.0 and g = Array.make n 0.0 in
    fill p 0.5 31;
    Adam.reset ();
    for t = 1 to steps do
      fill g 1.0 (1000 + t);
      Adam.begin_step_pow ();
      Adam.update_block p g 0 n
    done;
    let oc = open_out Sys.argv.(2) in
    Printf.fprintf oc "n %d steps %d\n" n steps;
    Array.iter (fun x -> Printf.fprintf oc "%.17g\n" x) p;
    close_out oc;
    Printf.printf "   AD3: %d updates over %d parameters -> %s\n" steps n Sys.argv.(2)
  end else begin
    (* AD6: 52's forward, 52's backward, and one Adam update, and every parameter must move. *)
    fill Gru_deep.w_ih0 0.3 1;  fill Gru_deep.w_hh0 0.3 2;
    fill Gru_deep.b_ih0 0.2 3;  fill Gru_deep.b_hh0 0.2 4;
    fill Gru_deep.w_ih1 0.2 11; fill Gru_deep.w_hh1 0.2 12;
    fill Gru_deep.b_ih1 0.1 13; fill Gru_deep.b_hh1 0.1 14;
    fill Gru_deep.head_w 0.1 21; fill Gru_deep.head_b 0.1 22;
    fill Gru_deep.x_seq 1.0 5;
    fill Gru_deep.h_init0 1.0 6; fill Gru_deep.h_init1 1.0 16;
    fill Gru_deep.coeff_h 1.0 7; fill Gru_deep.coeff_y 1.0 17;
    let blocks = [|
      ("w_ih0", Gru_deep.w_ih0, Gru_deep.g_w_ih0);
      ("w_hh0", Gru_deep.w_hh0, Gru_deep.g_w_hh0);
      ("b_ih0", Gru_deep.b_ih0, Gru_deep.g_b_ih0);
      ("b_hh0", Gru_deep.b_hh0, Gru_deep.g_b_hh0);
      ("w_ih1", Gru_deep.w_ih1, Gru_deep.g_w_ih1);
      ("w_hh1", Gru_deep.w_hh1, Gru_deep.g_w_hh1);
      ("b_ih1", Gru_deep.b_ih1, Gru_deep.g_b_ih1);
      ("b_hh1", Gru_deep.b_hh1, Gru_deep.g_b_hh1);
      ("head_w", Gru_deep.head_w, Gru_deep.g_head_w);
      ("head_b", Gru_deep.head_b, Gru_deep.g_head_b);
    |] in
    let before = Array.map (fun (_, p, _) -> Array.copy p) blocks in
    Gru_deep.forward_deep 250;
    let l0 = Gru_deep.loss_out.(0) in
    Gru_deep.backward_deep 250;
    Adam.reset ();
    Adam.begin_step_pow ();
    let off = ref 0 in
    Array.iter (fun (_, p, g) ->
      Adam.update_block p g !off (Array.length p);
      off := !off + Array.length p) blocks;
    Gru_deep.forward_deep 250;
    let l1 = Gru_deep.loss_out.(0) in
    Printf.printf "== Section 53, AD6: one full training step ==\n";
    Printf.printf "   parameters in the update %d\n" !off;
    Printf.printf "   loss before %.9e   after %.9e   change %+.3e\n" l0 l1 (l1 -. l0);
    let total_unmoved = ref 0 in
    Array.iteri (fun k (nm, p, _) ->
      let un = ref 0 in
      Array.iteri (fun i x -> if x = before.(k).(i) then incr un) p;
      total_unmoved := !total_unmoved + !un;
      if !un > 0 then Printf.printf "   UNMOVED %-7s %d of %d\n" nm !un (Array.length p))
      blocks;
    Printf.printf "   unmoved %d of %d\n" !total_unmoved !off;
    exit (if !total_unmoved = 0 then 0 else if !total_unmoved <= 100 then 1 else 2)
  end
