# Fix the `(host_offset, length)` semantics mismatch behind the golden-filed "Partial data integrity verification: FAIL"

## Goal

`test/test_memory.expected` records `Partial data integrity verification: FAIL`
(under `=== Partial Transfer Tests ===`). `dune runtest` is green only because
the golden file encodes the FAIL — a golden-filed failure silently masks any
future regression in that data-integrity check.

Investigation (static analysis, confirmed against the current tree) found the
FAIL to be a **genuine bug**: `memcpy_H_to_D_impl` and `memcpy_D_to_H_impl` in
`src/cuda.ml` disagree on what the `(host_offset, length)` pair means. The
H-to-D side reads `length` as an element *count* (with `host_offset` shifting
only the host pointer); the D-to-H side reads `length` as an *end index*,
copying `length - offset` elements. The same `(host_offset, length)` pair
therefore produces two incompatible byte counts, and the round-trip in
`test_partial_transfers` short-copies the read-back so the integrity loop
mismatches.

The user resolved the semantics question (2026-06-23): **`length` is the element
COUNT to copy; the copy spans host elements `[host_offset, host_offset + length)`.**
This confirms the H-to-D reading as the intended contract. The fix is to make
D-to-H match it, document the contract, and regenerate the golden file so the
line reads `PASS` rather than masking the check.

This is not a hardware/driver limitation and not a test artifact: it predates
task-55a11fa3 / task-66a3bbff (introduced with the test suite in `cdb099c`; the
device-offset work `bdf36b7` left both size formulas untouched), so it is
independent of the device-region/offset work.

## Acceptance Criteria

- [ ] **AC1 — D-to-H size formula matches H-to-D.** In `memcpy_D_to_H_impl`
  (`src/cuda.ml`), the `Some offset, Some length` branch computes the copy size
  as `elem_bytes * length` (element count), not `elem_bytes * (length - offset)`.
  After the change, for a given `(host_offset, length)` pair the H-to-D and D-to-H
  copies move the same number of bytes, and `host_offset` shifts only the host
  pointer on both sides. The single-site fix covers both the synchronous
  (`Deviceptr.memcpy_D_to_H`) and asynchronous (`Stream.memcpy_D_to_H`) paths,
  since both route through `memcpy_D_to_H_impl`.

- [ ] **AC2 — Contract documented in `src/cuda.mli`.** The doc comments on
  `memcpy_H_to_D` and `memcpy_D_to_H` (both the synchronous `Deviceptr.*` and the
  asynchronous `Stream.*` declarations) state the resolved contract: `length` is
  the number of elements copied, spanning host elements
  `[host_offset, host_offset + length)`, and `host_offset` shifts only the host
  pointer. The existing wording about `dst_offset` / `src_offset` (device-side
  byte offsets, and the "size reduced by the device offset when `length` is
  absent" behavior) is preserved.

- [ ] **AC3 — Golden file reflects a real PASS.** After the fix,
  `test/test_memory.expected` line `Partial data integrity verification: FAIL`
  reads `Partial data integrity verification: PASS`, and that value is produced
  by an actually-passing runtime check on a CUDA host (see AC5), not by hand-
  editing the golden file to mask a still-failing test.

- [ ] **AC4 — `dune build` and `dune runtest` are green on a CUDA host.** On a
  machine with a working CUDA runtime (the federation's nvidia worker,
  minipc-wsl), `dune build` succeeds and `dune runtest` passes with the
  regenerated `test/test_memory.expected`. Verification evidence: the
  `test_memory` output line reads `Partial data integrity verification: PASS`,
  and `dune runtest` exits 0.

  Verification reachability note: `~/ocaml-cudajit` on minipc-wsl lags
  `origin/main`; `git fetch` and fast-forward the default branch to the commit
  carrying this fix before building, then confirm with `git -C ~/ocaml-cudajit
  merge-base --is-ancestor <fix-commit> HEAD` so the green run is attributable to
  the fix rather than stale checkout state.

- [ ] **AC5 — Fallback only if PASS is genuinely unreachable.** If, and only if,
  the runtime check on minipc-wsl cannot be made to PASS with the corrected size
  formula, document the FAIL with a specific root-cause rationale at the call
  site in `test/test_memory.ml` and replace the bare golden FAIL with an
  assertion of the *reason* (so the check cannot silently mask a future
  regression). Static analysis strongly indicates a viable PASS, so this is a
  fallback, not the expected outcome.

## Context

### How the copy size is computed today (`src/cuda.ml`)

`memcpy_H_to_D_impl` — `host_offset` shifts the host pointer; with a `length`,
the byte count is the element count:

```
| None, None -> full_size - dst_offset
| Some _, None ->
    invalid_arg "Cudajit.memcpy_H_to_D: providing offset requires providing length"
| _, Some length -> elem_bytes * length
```

`memcpy_D_to_H_impl` — same intent, but the `Some offset, Some length` branch
subtracts the offset, treating `length` as an end index:

```
| None, None -> full_size - src_offset
| Some offset, None -> full_size - (elem_bytes * offset) - src_offset
| None, Some length -> elem_bytes * length
| Some offset, Some length -> elem_bytes * (length - offset)   (* <- the bug *)
```

Only the last branch is wrong under the resolved contract. The
`Some offset, None` branch (no explicit length → copy the remainder of the host
buffer from `host_offset` to its end) is consistent with "length = count" and
stays as-is; the `None, Some length` branch is already correct.

Both `Deviceptr.memcpy_D_to_H` and `Stream.memcpy_D_to_H` delegate to
`memcpy_D_to_H_impl`, so one edit fixes sync and async.

### Why the test FAILs today (`test/test_memory.ml`, `test_partial_transfers`)

With `offset = 100`, `length = 500`, `size = 1024`, host `float_array[i] = i mod 50`:

- H-to-D: `~host_offset:100 ~length:500` copies `float_array[100..599]` (500
  elements) into `dptr[0..1999 bytes]`.
- D-to-H (buggy): size `= 4 * (500 - 100) = 400` elements, copying `dptr[0..399]`
  into `result_array[100..499]`. `result_array[500..599]` is never written and
  stays at the fill value `-1.0`.
- The integrity loop checks `result_array[i] == float_array[i]` for
  `i ∈ [100, 599]` → mismatches on `[500, 599]` → `mismatch := true` → FAIL.

With the fixed formula (size `= 4 * 500` = 500 elements), D-to-H copies
`dptr[0..1999 bytes]` into `result_array[100..599]`, so
`result_array[i] = dptr[i-100] = float_array[i]` for `i ∈ [100, 599]`, and the
untransferred ranges `[0,99] ∪ [600,1023]` stay at `-1.0`. Both the integrity
loop and the untransferred-unchanged check pass — **no test change is required**;
the round-trip becomes internally consistent purely from the size fix.

### Docs to update (`src/cuda.mli`)

The four doc comments (sync `memcpy_H_to_D` / `memcpy_D_to_H` ≈ the "Copies the
bigarray (or its interval)…" blocks, and the two async `Stream.*` equivalents)
currently say only "[host_offset] and [length] are in numbers of elements" — they
do not disambiguate count-vs-end-index, which is what let the two impls drift.
They must state the count semantics and the `[host_offset, host_offset + length)`
span.

### Provenance / non-interaction

Introduced with the test suite (`cdb099c`); the device byte-offset work
(`bdf36b7`, "Add device-side byte offsets to cudajit copy primitives") left both
size formulas untouched. The bug is independent of the device-region/offset work
from task-55a11fa3 (PR #10/#11) and task-66a3bbff.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

1. In `src/cuda.ml`, change the `memcpy_D_to_H_impl` branch
   `| Some offset, Some length -> elem_bytes * (length - offset)` to
   `| Some _offset, Some length -> elem_bytes * length` (the `host_offset` already
   shifts the host destination pointer in the `host` computation below; it must
   not also reduce the byte count).
2. Update the four `cuda.mli` doc comments (sync + async, H-to-D + D-to-H) to
   state: `length` = number of elements copied, spanning host elements
   `[host_offset, host_offset + length)`; `host_offset` shifts only the host
   pointer. Preserve the existing device-offset wording.
3. Build and run `dune runtest` on minipc-wsl; the `Partial data integrity
   verification` line should now print `PASS`. Promote the corrected output into
   `test/test_memory.expected` (regenerate via `dune runtest --auto-promote`, or
   update the one line), and re-run `dune runtest` to confirm green.

## Scope

**In scope:**
- The `Some offset, Some length` size formula in `memcpy_D_to_H_impl`
  (`src/cuda.ml`).
- The `(host_offset, length)` contract docs on `memcpy_H_to_D` / `memcpy_D_to_H`
  (sync + async) in `src/cuda.mli`.
- Regenerating the one affected line in `test/test_memory.expected`.
- CUDA-host build + `dune runtest` confirmation on minipc-wsl (requires
  `gpu: nvidia`).

**Out of scope:**
- The H-to-D size formula and the D-to-H `Some offset, None` / `None, Some length`
  branches — already correct under the resolved contract.
- The device-side `dst_offset` / `src_offset` behavior and its "reduce size when
  `length` absent" logic — unchanged.
- Any broader redesign of the partial-transfer test beyond the single `.expected`
  line.

**Behavior-change note:** this changes the documented D-to-H contract for
callers that passed both `host_offset` and `length` and relied on the old
`length - offset` (end-index) reading. The only in-repo consumer is this test;
flag it as a minor contract change for any external caller.

**Dependencies:** none blocking. Relates to task-55a11fa3 (adjacent
device-offset work) but is independent of it.
