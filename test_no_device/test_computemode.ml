(* Regression test for UNCATEGORIZED compute-mode handling.
   Uses Cuda.Device.computemode_of_int to route raw integers through the full
   conversion chain (cu_computemode_of_int -> computemode_of_cu), then checks
   sexp_of_computemode output.  No GPU device or Cu.init() required.

   Enforced two ways:
   1. Diff against test_computemode.expected (Dune rule) — catches printer regressions.
   2. check exits non-zero on mismatch — catches conversion regressions even if diff
      is bypassed (e.g. executable crashes before printing). *)

let check label expected sexp =
  let got = Sexplib0.Sexp.to_string sexp in
  if got = expected then Printf.printf "%s: PASS\n" label
  else begin
    Printf.eprintf "%s: FAIL (expected %S, got %S)\n" label expected got;
    exit 1
  end

let () =
  check "DEFAULT"
    "DEFAULT"
    (Cuda.Device.sexp_of_computemode (Cuda.Device.computemode_of_int 0));
  check "PROHIBITED"
    "PROHIBITED"
    (Cuda.Device.sexp_of_computemode (Cuda.Device.computemode_of_int 2));
  check "EXCLUSIVE_PROCESS"
    "EXCLUSIVE_PROCESS"
    (Cuda.Device.sexp_of_computemode (Cuda.Device.computemode_of_int 3));
  (* Value 1 = legacy CU_COMPUTEMODE_EXCLUSIVE removed in CUDA 8.0.
     computemode_of_cu previously raised invalid_arg on this arm (src/cuda.ml:236).
     If that line is reverted, computemode_of_int 1 raises an unhandled exception,
     the process exits non-zero, and the with-stdout-to rule fails before diff runs. *)
  check "UNCATEGORIZED 1"
    "(UNCATEGORIZED 1)"
    (Cuda.Device.sexp_of_computemode (Cuda.Device.computemode_of_int 1));
  check "UNCATEGORIZED 42"
    "(UNCATEGORIZED 42)"
    (Cuda.Device.sexp_of_computemode (Cuda.Device.computemode_of_int 42))
