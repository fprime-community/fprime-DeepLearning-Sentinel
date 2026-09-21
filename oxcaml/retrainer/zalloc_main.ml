(* E3's arithmetic check. docs/MODELS.md 47.9 X7.
 *
 * (!) A ZERO-ALLOC FUNCTION THAT COMPUTES NOTHING WOULD PASS THE CHECKER TRIVIALLY, so
 * this runs the annotated functions and checks their output against values worked out by
 * hand. X7 says "on real arithmetic"; this is what entitles the section to that phrase.
 *
 * This module is NOT annotated and is not part of the claim -- only Zalloc's four
 * functions are. *)

let ok = ref true

let check name got want =
  let good = Float.abs (got -. want) < 1e-12 in
  if not good then ok := false;
  Printf.printf "    %-22s got %.6f  want %.6f   %s\n" name got want
    (if good then "OK" else "MISMATCH")

let () =
  let n = Zalloc.n in
  Printf.printf "== E3: the annotated functions compute what they claim ==\n";

  (* matmul: a and b all ones, so every element of c is n. *)
  for i = 0 to (n * n) - 1 do
    Zalloc.a.(i) <- 1.0;
    Zalloc.b.(i) <- 1.0
  done;
  Zalloc.matmul ();
  check "c[0][0]" Zalloc.c.(0) (float_of_int n);
  check "c[last]" Zalloc.c.((n * n) - 1) (float_of_int n);

  (* grad step: w = 0, x = 1, y = -1, lr = 0.5.
     forward -> pred = 0;  d = 0 - (-1) = 1;  grad = 2;
     w <- 0 - 0.5 * 2 * 1 = -1.0 everywhere. *)
  for i = 0 to (n * n) - 1 do Zalloc.w.(i) <- 0.0 done;
  for i = 0 to n - 1 do
    Zalloc.x.(i) <- 1.0;
    Zalloc.y.(i) <- -1.0
  done;
  Zalloc.grad_step 0.5;
  check "pred[0] before step" Zalloc.pred.(0) 0.0;
  check "grad[0]" Zalloc.grad.(0) 2.0;
  check "w[0] after step" Zalloc.w.(0) (-1.0);
  check "w[last] after step" Zalloc.w.((n * n) - 1) (-1.0);

  (* a second step must move w again, so state really is carried. *)
  Zalloc.grad_step 0.5;
  check "w[0] after 2 steps" Zalloc.w.(0) (-1.0 -. (2.0 *. (Zalloc.pred.(0) -. (-1.0))
                                                    *. 0.5 /. 2.0 *. 2.0));

  Zalloc.checksum ();
  Printf.printf "    checksum slot         %.6f\n" Zalloc.out.(0);
  Printf.printf "== E3 arithmetic: %s ==\n" (if !ok then "all checks passed" else "FAILED");
  exit (if !ok then 0 else 1)
