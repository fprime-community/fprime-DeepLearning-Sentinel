(* Section 55 arm B's driver: DT3, DT4, DT5, DT7. Not annotated and not part of
 * the claim -- 49's gru_check.ml set that precedent.
 *
 * The digest is a CRC32 over the RAW BIT PATTERNS of every parameter, not over
 * their values. DeterminismTest.cpp:28-31 gives the reason: two floats that
 * compare equal can differ in their encoding, and the claim is about the
 * encoding. `Int32.bits_of_float` on a widened float32 is exact, because the
 * widening was. *)

let table =
  let t = Array.make 256 0l in
  for n = 0 to 255 do
    let c = ref (Int32.of_int n) in
    for _ = 0 to 7 do
      c := if Int32.logand !c 1l <> 0l
           then Int32.logxor 0xEDB88320l (Int32.shift_right_logical !c 1)
           else Int32.shift_right_logical !c 1
    done;
    t.(n) <- !c
  done; t

let crc_byte acc b =
  Int32.logxor (table.((Int32.to_int (Int32.logxor acc (Int32.of_int b))) land 0xFF))
               (Int32.shift_right_logical acc 8)

let crc_f32 acc v =
  let bits = Int32.bits_of_float v in
  let acc = ref acc in
  for s = 0 to 3 do
    acc := crc_byte !acc
             ((Int32.to_int (Int32.shift_right_logical bits (8 * s))) land 0xFF)
  done; !acc

let digest () =
  let acc = ref (Int32.lognot 0l) in
  let over (a : Gru_f32.f32) =
    for i = 0 to Bigarray.Array1.dim a - 1 do
      acc := crc_f32 !acc (Bigarray.Array1.unsafe_get a i)
    done in
  over Gru_f32.w_ih; over Gru_f32.w_hh; over Gru_f32.b_ih; over Gru_f32.b_hh;
  Int32.logand (Int32.lognot !acc) 0xFFFFFFFFl

(* A fixed, reproducible starting model. Not a fit -- a starting point. *)
let seed_parameters () =
  let fill (a : Gru_f32.f32) salt =
    for i = 0 to Bigarray.Array1.dim a - 1 do
      let m = ((i * 2654435761) + (salt * 40503)) land 0xFFFF in
      Bigarray.Array1.unsafe_set a i ((float_of_int m /. 65535.0 -. 0.5) *. 0.1)
    done in
  fill Gru_f32.w_ih 1; fill Gru_f32.w_hh 2;
  fill Gru_f32.b_ih 3; fill Gru_f32.b_hh 4;
  for i = 0 to 79 do Bigarray.Array1.unsafe_set Gru_f32.h i 0.0 done

let show label d steps masked =
  Printf.printf "   %-34s digest %08lx   steps %d   masked %d\n"
    label d steps masked

let () =
  let path = if Array.length Sys.argv > 1 then Sys.argv.(1) else "cycle55.crc" in

  (* DT4: two training sets differing by far more than 10x, same budget. *)
  seed_parameters ();
  Cycle55.run_cycle 7 12345;
  let d_small = digest () in
  let s_small = Cycle55.steps_taken.(0) and m_small = Cycle55.masked_units.(0) in
  show "set of 7" d_small s_small m_small;

  seed_parameters ();
  Cycle55.run_cycle 2048 12345;
  let d_large = digest () in
  let s_large = Cycle55.steps_taken.(0) and m_large = Cycle55.masked_units.(0) in
  show "set of 2048 (292x larger)" d_large s_large m_large;

  Printf.printf "   DT4  steps %d and %d against a budget of %d: %s\n"
    s_small s_large Cycle55.budget
    (if s_small = Cycle55.budget && s_large = Cycle55.budget then "EXACT" else "DIFFER");

  (* DT5: the mask sequence is a pure function of seed and counters. Drawn twice,
     and drawn again from a resumed start, without running the cycle. *)
  let draw () =
    let acc = ref 0 in
    for k = 0 to Cycle55.budget - 1 do
      for i = 0 to 79 do
        if Cycle55.keep 12345 k i then acc := (!acc * 31 + k + i) land 0xFFFFFF
      done
    done; !acc in
  let a = draw () and b = draw () in
  Printf.printf "   DT5  mask fingerprint %06x, redrawn %06x: %s\n"
    a b (if a = b then "identical" else "DIFFER");
  Printf.printf "   DT5  a different seed gives a different sequence: %s\n"
    (let c = (let acc = ref 0 in
              for k = 0 to Cycle55.budget - 1 do
                for i = 0 to 79 do
                  if Cycle55.keep 999 k i then acc := (!acc * 31 + k + i) land 0xFFFFFF
                done done; !acc) in
     if c <> a then "yes" else "NO -- the seed is not reaching the draw");

  (* DT3: across two process invocations. First writes, second compares. *)
  seed_parameters ();
  Cycle55.run_cycle 7 12345;
  let d = digest () in
  if Sys.file_exists path then begin
    let ic = open_in path in
    let previous = input_line ic in
    close_in ic;
    let now = Printf.sprintf "%08lx" d in
    Printf.printf "   DT3  across processes: %s == %s: %s\n" previous now
      (if previous = now then "IDENTICAL" else "DIFFER");
    if previous <> now then exit 1
  end else begin
    let oc = open_out path in
    output_string oc (Printf.sprintf "%08lx\n" d);
    close_out oc;
    Printf.printf "   DT3  first run: digest %08lx recorded for the next process\n" d
  end
