(* Section 59's driver. Not annotated and not part of the claim. *)

let read_into path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in
  really_input ic b 0 n; close_in ic;
  for i = 0 to n - 1 do
    Bigarray.Array1.unsafe_set Shadow59.file i (Char.code (Bytes.get b i))
  done; n

let write_out path n =
  let b = Bytes.create n in
  for i = 0 to n - 1 do
    Bytes.set b i (Char.chr (Bigarray.Array1.unsafe_get Shadow59.file i))
  done;
  let oc = open_out_bin path in output_bytes oc b; close_out oc

let () =
  let src = Sys.argv.(1) and dst = Sys.argv.(2) in
  let n = read_into src in
  let cb = Shadow59.get_u32 Shadow59.off_channels_bytes in
  let wb = Shadow59.get_u32 Shadow59.off_weights_bytes in
  let nw = wb / 4 in
  Printf.printf "   flying file %s: %d B, channels %d B, weights %d B (%d float32)\n"
    (Filename.basename src) n cb wb nw;

  (* A trained shadow: the flying weights nudged by a fixed, reproducible delta.
     Not a fit -- E5-b trains. This rung is about the WRITE. *)
  let w : Shadow59.f32 =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout nw in
  let w_off = 64 + cb in
  for i = 0 to nw - 1 do
    let bits = ref 0 in
    for s = 0 to 3 do
      bits := !bits lor ((Bigarray.Array1.unsafe_get Shadow59.file (w_off + (4 * i) + s)) lsl (8 * s))
    done;
    let v = Int32.float_of_bits (Int32.of_int (if !bits land 0x80000000 <> 0
                                               then !bits - (1 lsl 32) else !bits)) in
    Bigarray.Array1.unsafe_set w i (v +. 0.001)
  done;

  let rc = Shadow59.write_shadow w nw in
  Printf.printf "   write_shadow returned %d\n" rc;
  if rc <> 0 then exit 1;
  write_out dst n;
  Printf.printf "   shadow written to %s, %d B\n" (Filename.basename dst) n
