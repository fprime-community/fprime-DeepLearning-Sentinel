(* Section 63, OW4: does the unboxed `checksum` return what the slot form writes?
 *
 * The compile is half the prediction; OW4's HOLD band also requires the same value. A
 * function that holds `strict` and computes something else has not replaced anything.
 * Both forms are driven from the same arrays in one process, so the comparison is
 * bit-exact by construction rather than by transcription. *)

module Z = Zalloc_u
module F = Stdlib_upstream_compatible.Float_u

(* The project's LCG, as `scripts/s53_adam_reference.py:17-24` and the OCaml sides use:
   63-bit ints, so no intermediate overflow. *)
let fill arr scale off =
  let s = ref (12345 + off) in
  for i = 0 to Array.length arr - 1 do
    s := ((!s * 1103515245) + 12345) land 0x3FFFFFFF;
    arr.(i) <- scale *. ((float_of_int (!s mod 2000) /. 1000.0) -. 1.0)
  done

let () =
  fill Z.c 1.0 7;
  fill Z.w 1.0 91;

  Z.checksum_slot ();
  let slot = Z.out.(0) in
  let unboxed = F.to_float (Z.checksum_unboxed ()) in

  Printf.printf "-- OW4: the unboxed checksum against E3's slot form\n";
  Printf.printf "   slot form     %.17g\n" slot;
  Printf.printf "   unboxed form  %.17g\n" unboxed;
  let same = Int64.equal (Int64.bits_of_float slot) (Int64.bits_of_float unboxed) in
  Printf.printf "   bit-identical: %b\n" same;
  Printf.printf "   returns its float rather than writing a slot: true\n";
  if not same then exit 2
