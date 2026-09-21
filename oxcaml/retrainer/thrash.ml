(* E2 apparatus: a garbage collector under deliberate abuse. docs/MODELS.md 47.9.1.

   (!) THIS IS A SEPARATE MODULE FROM retrainer.ml ON PURPOSE. E1 pinned a five-function
   surface -- init, feed, step, status, export -- and 47.13.1 reports against it. Adding a
   sixth entry point there would quietly change the thing E1 measured. Nothing here is part
   of the retrainer's interface and nothing here would exist in a flight build.

   What it does: allocates as hard as it can, in shapes that force minor collections --
   short-lived arrays and lists, immediately dropped. Sys.opaque_identity stops the
   optimiser deleting the allocation it is the entire point of. *)

(* (!) THREE ALERTS ARE DISABLED HERE AND EVERY ONE OF THEM IS A FINDING, NOT A NUISANCE.
   OxCaml's stdlib refuses `Domain.spawn` three separate ways:

     unstable              "The Domain interface may change in incompatible ways in
                           the future."
     do_not_spawn_domains  "User programs should never spawn domains. To execute a
                           function on a domain, use [Multicore] from the threading
                           library. This is because spawning more than
                           [recommended_domain_count] domains (the CPU core count)
                           will significantly degrade GC performance."
     unsafe_multidomain    "Use [Domain.Safe.spawn]."

   The second one is the interesting one: **the language's own standard library tells
   user programs not to do the thing arm C exists to measure**, and says that exceeding
   the core count degrades GC performance -- which is what arm B2 deliberately does.

   They are disabled because THIS IS THE ADVERSARY, not the design. E2's job is to create
   the worst case a retrainer could inflict on a detector and see whether a process
   boundary contains it; apparatus that politely declined to stress the collector would
   measure nothing. Nothing in `retrainer.ml` spawns a domain, and nothing in a flight
   build would.

   Recorded at docs/MODELS.md 47.13.2. *)
[@@@alert "-unstable"]
[@@@alert "-do_not_spawn_domains"]
[@@@alert "-unsafe_multidomain"]

let running = Atomic.make true
let counter = Atomic.make 0

(* One unit of abuse. Sized so the minor heap turns over quickly rather than so the
   work is realistic: E2 asks what a GC does to a neighbour, not what a trainer does. *)
let churn () =
  let a = Array.make 512 0.0 in
  for i = 0 to 511 do
    a.(i) <- float_of_int i
  done;
  ignore (Sys.opaque_identity a);
  let l = List.init 64 (fun i -> (i, float_of_int i)) in
  ignore (Sys.opaque_identity l)

let burn () =
  while Atomic.get running do
    churn ();
    Atomic.incr counter
  done

let handle : unit Domain.t option ref = ref None

(* Arm C: a second domain IN THIS PROCESS, stopped by the caller. *)
let start_domain () =
  Atomic.set running true;
  Atomic.set counter 0;
  handle := Some (Domain.spawn burn);
  0

let stop () =
  Atomic.set running false;
  (match !handle with Some d -> Domain.join d | None -> ());
  handle := None;
  Atomic.get counter

(* Arms B and B2: n domains for a bounded wall-clock time, then the REAL counter.
   Bounded rather than signal-terminated because a signal handler cannot safely call
   into OCaml, and 47.9.2 requires each arm to report the thrasher's own progress --
   a proxy for "the child was running" is exactly the weak evidence that rider exists
   to forbid. *)
let run_for (n : int) (ms : int) (path : string) =
  Atomic.set running true;
  Atomic.set counter 0;
  let ds = List.init n (fun _ -> Domain.spawn burn) in
  (* Checkpoint the counter every 100 ms. The parent kills this process when its
     measurement block ends, at a moment this process cannot predict, so a counter
     written only at exit would never be written at all -- and 47.9.2 requires a REAL
     progress figure per arm, not a proxy for the child having been alive. *)
  let slices = (ms + 99) / 100 in
  (try
     for _ = 1 to slices do
       Unix.sleepf 0.1;
       let oc = open_out path in
       output_string oc (string_of_int (Atomic.get counter));
       output_char oc '\n';
       close_out oc
     done
   with _ -> ());
  Atomic.set running false;
  List.iter Domain.join ds;
  Atomic.get counter

let count () = Atomic.get counter

(* (!) The multidomain alert is LIVE here, unlike in retrainer.ml, because this module
   really does spawn domains. It is still the registration that is flagged, and the
   registration is safe for a reason worth stating: the five closures below are looked up
   and called ONLY from the main thread, by C, and the function that runs on a spawned
   domain -- `burn` -- is not registered and is never looked up across a domain boundary.
   `Callback.Safe.register` would demand `'a @ portable`, which a closure over an
   `Atomic.t` ref cell is not, and restructuring E2's apparatus around OxCaml's modes would
   be work in service of the measuring instrument rather than the measurement. *)
let[@alert "-unsafe_multidomain"] () =
  Callback.register "sentinel_thrash_start_domain" start_domain;
  Callback.register "sentinel_thrash_run_for" (fun (n, ms, path) -> run_for n ms path);
  Callback.register "sentinel_thrash_stop" stop;
  Callback.register "sentinel_thrash_count" count
