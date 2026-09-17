(* Section 56's scorer. Not annotated and not part of the claim.
 *
 * (!) THE SCORER KNOWS GROUND TRUTH AND THE SELECTOR DOES NOT. Stop 25. The split
 * between this file and window56.ml is the enforcement: everything below about
 * seeds, fault times and healthy controls lives here, and Window56.push takes two
 * bits. *)

let fault_start = 8000          (* 42.8.2: 19,225 - 11,225 in-limits ticks *)

type trace = { lim : int array; emit : int array }

let read path =
  let ic = open_in path in
  let _ = input_line ic in                       (* header *)
  let lim = ref [] and emit = ref [] in
  (try
     while true do
       let line = input_line ic in
       match String.split_on_char ',' line with
       | _tick :: _wall :: e :: _fused :: _peak :: la :: _ ->
         emit := (if e = "1" then 1 else 0) :: !emit;
         lim := (if int_of_string la >= 0 then 1 else 0) :: !lim
       | _ -> ()
     done
   with End_of_file -> close_in ic);
  { lim = Array.of_list (List.rev !lim); emit = Array.of_list (List.rev !emit) }

(* Run the selector over a trace and return the per-tick verdicts of one gate set. *)
let verdicts t which n =
  Window56.reset ();
  let out = Array.make n 0 in
  for i = 0 to n - 1 do
    Window56.push t.lim.(i) t.emit.(i);
    out.(i) <- (match which with
        | `All -> Window56.admits_all ()
        | `Rate -> Window56.admits_rate_only ()
        | `Quiet -> Window56.admits_quiet_only ())
  done;
  out

let count_from v lo = 
  let c = ref 0 in
  for i = lo to Array.length v - 1 do if v.(i) = 1 then incr c done; !c

let () =
  let dir = if Array.length Sys.argv > 1 then Sys.argv.(1) else "runs/testbed" in

  (* WS1 is the build. WS6: a candidate shorter than the floor admits nothing. *)
  Window56.reset ();
  for _ = 1 to Window56.w - 1 do Window56.push 0 0 done;
  let short = Window56.admits_all () and short_suff = Window56.sufficient () in
  Window56.push 0 0;
  let exact = Window56.admits_all () in
  Printf.printf "   WS6  at W-1 ticks admits %d (sufficient %d); at W ticks admits %d: %s\n"
    short short_suff exact
    (if short = 0 && short_suff = 0 && exact = 1 then "REFUSES then ADMITS" else "UNEXPECTED");

  (* WS3: the healthy controls. *)
  Printf.printf "\n   WS3  healthy controls, ticks admitted by the three gates\n";
  let healthy_ok = ref true in
  for s = 1 to 10 do
    let t = read (Printf.sprintf "%s/h_%d.csv" dir s) in
    let n = Array.length t.lim in
    let v = verdicts t `All n in
    let a = count_from v 0 in
    if a < Window56.w then healthy_ok := false;
    Printf.printf "     h_%-2d  %5d of %5d admitted\n" s a n
  done;
  Printf.printf "   WS3  every control admits at least the floor (%d): %s\n"
    Window56.w (if !healthy_ok then "YES" else "NO");

  (* WS4 and WS5: the seeded runs, contaminated ticks only. *)
  Printf.printf "\n   WS4/WS5  seeded runs, CONTAMINATED ticks (>= %d) admitted\n" fault_start;
  Printf.printf "     %-6s %8s %8s %8s\n" "run" "3 gates" "rate-only" "quiet-only";
  let ws4 = ref true and ws5 = ref true in
  for s = 1 to 10 do
    let t = read (Printf.sprintf "%s/f_%d.csv" dir s) in
    let n = Array.length t.lim in
    let a = count_from (verdicts t `All n) fault_start in
    let r = count_from (verdicts t `Rate n) fault_start in
    let q = count_from (verdicts t `Quiet n) fault_start in
    if not (a < r) then ws4 := false;
    if not (q > 5000) then ws5 := false;
    Printf.printf "     f_%-4d %8d %8d %8d\n" s a r q
  done;
  Printf.printf "   WS4  three gates strictly fewer than rate-only on all ten: %s\n"
    (if !ws4 then "YES" else "NO");
  Printf.printf "   WS5  quiet-only admits > 5,000 contaminated on all ten: %s\n"
    (if !ws5 then "YES" else "NO");

  (* WS2: causality by prefix. *)
  let t = read (Printf.sprintf "%s/f_1.csv" dir) in
  let n = Array.length t.lim in
  let full = verdicts t `All n in
  let ok = ref true in
  List.iter (fun l ->
      let pre = verdicts t `All l in
      for i = 0 to l - 1 do if pre.(i) <> full.(i) then ok := false done)
    [1000; 7000; 12000; 19000];
  Printf.printf "\n   WS2  prefixes 1000/7000/12000/19000 agree with the full run: %s\n"
    (if !ok then "YES" else "NO")
