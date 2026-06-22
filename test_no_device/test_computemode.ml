(* Regression test for UNCATEGORIZED compute-mode handling.
   Exercises Cuda.Device.sexp_of_computemode on all constructors including
   UNCATEGORIZED, without requiring a GPU device or calling Cu.init (). *)

let check label expected sexp =
  let got = Sexplib0.Sexp.to_string sexp in
  if got = expected then Printf.printf "%s: PASS\n" label
  else Printf.printf "%s: FAIL (expected %S, got %S)\n" label expected got

let () =
  check "DEFAULT"          "DEFAULT"          (Cuda.Device.sexp_of_computemode DEFAULT);
  check "PROHIBITED"       "PROHIBITED"       (Cuda.Device.sexp_of_computemode PROHIBITED);
  check "EXCLUSIVE_PROCESS""EXCLUSIVE_PROCESS"(Cuda.Device.sexp_of_computemode EXCLUSIVE_PROCESS);
  (* Value 1 was the legacy CU_COMPUTEMODE_EXCLUSIVE removed in CUDA 8.0. Previously
     computemode_of_cu raised invalid_arg on this; now it surfaces as UNCATEGORIZED 1. *)
  check "UNCATEGORIZED 1"  "(UNCATEGORIZED 1)" (Cuda.Device.sexp_of_computemode (UNCATEGORIZED 1L));
  check "UNCATEGORIZED 42" "(UNCATEGORIZED 42)"(Cuda.Device.sexp_of_computemode (UNCATEGORIZED 42L))
