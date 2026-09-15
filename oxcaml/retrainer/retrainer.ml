(* E1, the OCaml half of the pipe. No ML, and deliberately none: E1 proves the chain
   -- compile, link, runtime startup, boundary, F' integration -- and arithmetic whose
   answer can be checked by hand is the right payload for that. See docs/MODELS.md 47.9,
   predictions X1 to X3.

   (!) EVERY EXCEPTION IS CAUGHT HERE AND RETURNED AS A STATUS CODE. F' forbids
   exceptions (CPP-25) and the C++ side builds -fno-exceptions, so an exception reaching
   the boundary is undefined behaviour. docs/MODELS.md 47.6 requires that to be
   structurally impossible rather than merely avoided, so every registered callback is
   wrapped in `guard` and there is no path out of this module that raises.

   The accumulator is the shape flight/src/TrailingWindow.cpp already uses -- F64 sum,
   sum of squares and a count over a ring, never differenced float32 prefix sums, which
   is D37's defect. Reused as a shape, not as code. *)

(* Status codes. Mirror sentinel_retrainer.h; the C side never sees anything else. *)
let ok                = 0
let err_not_init      = 1
let err_already_init  = 2
let err_capacity      = 3
let err_overflow      = 4
let err_short_buffer  = 5
let err_internal      = 6

(* State codes returned by `status`. *)
let state_uninit   = 0
let state_ready    = 1
let state_fed      = 2
let state_stepped  = 3

type acc = {
  capacity   : int;
  mutable n  : int;
  mutable sum   : float;
  mutable sumsq : float;
  mutable state : int;
}

let store : acc option ref = ref None

(* (!) The only place an exception may be turned into a status. Nothing below raises
   past this. `Out_of_memory` and `Stack_overflow` are caught too: a status code the
   caller can act on beats undefined behaviour at a -fno-exceptions boundary. *)
let guard f = try f () with _ -> err_internal

let init capacity =
  guard (fun () ->
    if capacity <= 0 then err_capacity
    else match !store with
      | Some _ -> err_already_init
      | None ->
        store := Some { capacity; n = 0; sum = 0.0; sumsq = 0.0; state = state_ready };
        ok)

(* Telemetry arrives as a Bigarray. Its buffer lives OUTSIDE the OCaml heap and is
   therefore never moved by the GC, which is the entire reason it is the right carrier
   at this boundary (docs/MODELS.md 47.6, boundary B). No OCaml value derived from it
   is retained after this call returns. *)
let feed (samples : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t) =
  guard (fun () ->
    match !store with
    | None -> err_not_init
    | Some a ->
      let k = Bigarray.Array1.dim samples in
      if a.n + k > a.capacity then err_overflow
      else begin
        for i = 0 to k - 1 do
          let x = Bigarray.Array1.unsafe_get samples i in
          a.sum   <- a.sum +. x;
          a.sumsq <- a.sumsq +. (x *. x)
        done;
        a.n <- a.n + k;
        a.state <- state_fed;
        ok
      end)

let step () =
  guard (fun () ->
    match !store with
    | None -> err_not_init
    | Some a -> a.state <- state_stepped; ok)

let status () =
  guard (fun () -> match !store with None -> state_uninit | Some a -> a.state)

(* Writes [| count; sum; mean |] into a caller-owned Bigarray. The caller owns the
   memory; this function retains nothing. *)
let export (out : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t) =
  guard (fun () ->
    match !store with
    | None -> err_not_init
    | Some a ->
      if Bigarray.Array1.dim out < 3 then err_short_buffer
      else begin
        let n = float_of_int a.n in
        Bigarray.Array1.unsafe_set out 0 n;
        Bigarray.Array1.unsafe_set out 1 a.sum;
        Bigarray.Array1.unsafe_set out 2 (if a.n = 0 then 0.0 else a.sum /. n);
        ok
      end)

(* (!) OxCaml's stdlib flags `Callback.register` as multidomain-unsafe and points at
   `Callback.Safe.register`, whose signature demands `'a @ portable`. A closure over a
   mutable `ref` is NOT portable, so taking the safe variant would mean restructuring this
   module around OxCaml's modes -- work that belongs to E5 and not to a pipe with no ML in
   it.

   The alert is disabled HERE, narrowly, and only because the thing it warns about cannot
   happen by construction: this process is SINGLE-DOMAIN. That is not a convenience, it is
   `docs/DECISIONS.md` D70 consequence 2 -- the retrainer is a separate OS process
   precisely so that no second domain shares a collector with the detector. If that ever
   stops being true, this opt-out is wrong and the alert should be allowed to fire.

   Everything else is built at `-warn-error +a -alert @all`, so this is the one exception
   and any other alert is a build failure. *)
let[@alert "-unsafe_multidomain"] () =
  Callback.register "sentinel_rt_init"   init;
  Callback.register "sentinel_rt_feed"   feed;
  Callback.register "sentinel_rt_step"   step;
  Callback.register "sentinel_rt_status" status;
  Callback.register "sentinel_rt_export" export
