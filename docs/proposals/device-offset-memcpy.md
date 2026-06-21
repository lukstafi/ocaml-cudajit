# Add device-side byte offset to cudajit copy primitives

## Goal

OCANNL's Universal Pool Allocator (gh-ocannl-344) backs many tnodes with one
slab allocation and addresses each sub-region at a distinct byte offset. On
CPU/Metal this is base+offset pointer arithmetic; on CUDA it is **not
expressible** against cudajit's public API, because `Deviceptr.t` is abstract
and owning (`type deviceptr = Deviceptr of { ptr : memptr; freed : atomic_bool }`)
with no public constructor, offset, or raw-address accessor. So a slab
sub-region pointer — base + byte offset, which must NOT be individually freed —
has no representation, and copying into/out of a slab sub-region (e.g.
`memcpy_D_to_D` between two pooled sub-regions) cannot be done.

This proposal adds the missing primitive so the OCANNL CUDA pool allocator can
be implemented (consumer rewrite tracked as task-6abfb6a9, which this task
**blocks**).

The chosen design (confirmed with the user, option B over an offset-pointer
sub-view) mirrors OCANNL's shipped Metal backend one-to-one: the slab base
`Deviceptr.t` is passed unchanged and the byte offset is applied **at the copy
operation**, exactly as `metal_backend.ml`'s `resolve_pool` ignores the offset
in the handle and applies it via blit source/destination offsets.

## Acceptance Criteria

1. A device-side byte offset is added to **all** of cudajit's device-targeting
   copy primitives, on both the synchronous (`Deviceptr` module) and
   asynchronous (`Stream` module) sides:
   - `memcpy_D_to_D` — independent offsets for both ends (a copy between two
     pooled sub-regions needs both a `dst` and a `src` offset).
   - `memcpy_H_to_D` — a device-side `dst` offset (the device end may target a
     slab sub-region even when the host-side offset is 0).
   - `memcpy_D_to_H` — a device-side `src` offset (symmetric to the above).
   - The async (`Stream`) variants of the three above.
2. The new offset parameters are **optional** and default to 0, so every
   existing call site compiles and behaves identically (the existing
   allocate/free path and all current copies are unchanged).
3. Offsets are expressed in **bytes** (matching the device-side granularity;
   the existing host-side `host_offset`/`length` element-count parameters are
   left as they are).
4. The device-side offset is applied to the size accounting consistently with
   the existing helpers: when a copy size is derived from the full allocation /
   bigarray rather than an explicit size, a non-zero device offset must not
   cause a copy to read or write past the end of the allocation. (The behaviour
   here should match how the existing `host_offset` reduces the copied size in
   `memcpy_D_to_H_impl`; see Context.)
5. `Deviceptr.t` stays **abstract and owning** — no offset-pointer sub-view, no
   public raw-pointer constructor or accessor is added.
6. `.mli` docstrings document the new parameters (bytes; device end only;
   default 0) and the existing `cuMemcpy*` reference links are retained.
7. Verified by a runtest on a real CUDA host (minipc-wsl) that copies into a
   **non-zero device byte offset** of a larger allocation and reads it back
   (via a complementary offset copy) to confirm the bytes land at the intended
   sub-region and neighbouring bytes are untouched. The existing test suite
   continues to pass.

   AC verification reachability: the build/test host is `minipc-wsl` (reachable
   by passwordless SSH from mac-studio); the local `~/ocaml-cudajit` clone is a
   normal git worktree, so the implementation diff is verifiable by
   `git -C /Users/lukstafi/ocaml-cudajit diff`/`show` on the merge commit. The
   runtime AC (item 7) is GPU-bound and is verified by the test's PASS output
   on minipc-wsl, not by a SHA.

## Context

All paths are in `ocaml-cudajit/src/cuda.ml` / `cuda.mli` unless noted.

**The raw device pointer is a `uint64`, so an offset is plain arithmetic.**
`type memptr = Unsigned.uint64` (cuda.ml) and the FFI type
`cu_deviceptr = typedef cu_deviceptr_v2 "CUdeviceptr"` is `uint64_t`
(`cuda_ffi/bindings_types.ml`). A device byte offset is therefore
`Unsigned.UInt64.add ptr (Unsigned.UInt64.of_int offset_bytes)` applied to the
`ptr` field unpacked from `Deviceptr { ptr; freed }` — no FFI/binding change is
needed; the offset never escapes as a new `Deviceptr.t`.

**Synchronous primitives** (`module Deviceptr`):
- `memcpy_H_to_D_unsafe`, `memcpy_H_to_D` (the latter via helper
  `memcpy_H_to_D_impl`).
- `memcpy_D_to_H_unsafe`, `memcpy_D_to_H` (via `memcpy_D_to_H_impl`).
- `memcpy_D_to_D` — unpacks both `dst` and `src` `ptr` fields and calls
  `Cuda.cu_memcpy_D_to_D dst src size`.

**Asynchronous primitives** (`module Stream`):
- `memcpy_H_to_D_unsafe` / `memcpy_H_to_D` (reuses the same `memcpy_H_to_D_impl`).
- `memcpy_D_to_H_unsafe` / `memcpy_D_to_H` (reuses `memcpy_D_to_H_impl`).
- `memcpy_D_to_D` calling `Cuda.cu_memcpy_D_to_D_async dst src size stream`.

**Shared size helpers** (cuda.ml, before the `Deviceptr` module):
- `memcpy_H_to_D_impl ?host_offset ?length ~dst ~src memcpy` — computes
  `size_in_bytes` from the bigarray (or interval) and a host pointer, then calls
  the passed `memcpy ~dst ~src:host ~size_in_bytes`. The device-side offset
  belongs on `dst` here; threading it through this helper keeps both the sync
  and async `memcpy_H_to_D` covered (they share the helper).
- `memcpy_D_to_H_impl ?host_offset ?length ~dst ~src memcpy` — symmetric; note
  it already *reduces* `size_in_bytes` when `host_offset` is given
  (`full_size - elem_bytes * offset`). The device-side offset belongs on `src`
  here; see AC item 4 for the analogous size guard.
- `get_size_in_bytes ?kind ?length ?size_in_bytes provenance` — used by
  `memcpy_D_to_D` (and `memcpy_peer`); the device offsets are applied to the
  pointers, not to this size computation, but a non-zero offset combined with a
  full-allocation size would overrun — the implementer should decide whether to
  validate or document this (the OCANNL caller always passes an explicit size).

**The Metal precedent this mirrors** (consumer side, not in this repo):
`arrayjit/lib/metal_backend.ml` `resolve_pool` returns the slab buffer handle
with the byte offset deliberately *not* folded in, and copies apply it via
`BlitCommandEncoder.copy_from_buffer ~source_offset ~destination_offset`. The
CUDA `cuda_backend` rewrite (task-6abfb6a9) becomes a structural copy of that
pattern once these offset parameters exist.

**`memcpy_peer`** is intentionally *out of scope* (see Scope) — the pool
allocator is single-device.

**Kernel-arg path needs no change**: a buffer passed as
`kernel_param = Tensor of Deviceptr.t` receives the slab *base*; the per-buffer
byte offset is folded into the generated kernel's addressing in OCANNL codegen
(again mirroring Metal), so no cudajit change is required for kernel arguments.

**Existing tests**: `test/test_memory.ml` exercises `alloc_and_memcpy`,
`memcpy_D_to_H`, and partial (host-offset) transfers with deterministic
patterns and PASS/FAIL prints; the new offset test should follow this style and
live alongside it (registered in `test/dune`).

## Approach

*Suggested approach — agents may deviate if they find a better path.*

The change is mechanical and local. For each device-targeting copy:

1. Add an optional byte-offset parameter, defaulting to 0, naming the device
   end being offset:
   - `memcpy_H_to_D`: `?dst_offset:int` (bytes) — sync and async.
   - `memcpy_D_to_H`: `?src_offset:int` (bytes) — sync and async.
   - `memcpy_D_to_D`: `?dst_offset:int -> ?src_offset:int` (bytes) — sync and
     async.
   - The `_unsafe` variants may take the offset too, or apply it before calling
     the binding; pick whichever keeps the diff smallest while keeping the
     public `_unsafe` signatures coherent.
2. Apply the offset by advancing the unpacked `ptr` with
   `Unsigned.UInt64.add ptr (Unsigned.UInt64.of_int offset)` immediately before
   the `Cuda.cu_memcpy_*` call. The `freed` flag and the `check_freed` guards
   stay on the original `Deviceptr.t`.
3. Thread `dst_offset` through `memcpy_H_to_D_impl` and `src_offset` through
   `memcpy_D_to_H_impl` so both the sync and async public functions are covered
   by the single shared helper (they already share it). Mind the
   size-reduction logic in `memcpy_D_to_H_impl` (AC item 4).
4. Update the `.mli` signatures and docstrings for all six functions (and the
   `_unsafe` ones if their signatures change).
5. Add a runtest to `test/test_memory.ml` (+ `.expected`, + `test/dune`
   registration): allocate a buffer larger than the payload, `memcpy_H_to_D`
   into a non-zero `dst_offset`, copy back with the matching `src_offset` via
   `memcpy_D_to_H`, and a `memcpy_D_to_D` between two non-zero offsets; verify
   the bytes land correctly and neighbouring bytes are untouched.

Parameter naming (`dst_offset`/`src_offset` in bytes) is a suggestion; the
implementer may prefer a single `?device_offset` where only one device end
exists. Keep it consistent across the sync/async pairs.

## Scope

**In scope**: device-side byte offsets on `memcpy_D_to_D`, `memcpy_H_to_D`,
`memcpy_D_to_H` and their async (`Stream`) variants; `.mli` docs; one runtest on
minipc-wsl.

**Out of scope**:
- `memcpy_peer` (cross-device; the pool allocator is single-device).
- Any offset-pointer / sub-view representation on `Deviceptr.t` (option A,
  explicitly rejected by the user in favour of option B).
- Re-exposing a raw `Deviceptr.t` constructor or `to_uint64`/`of_uint64`.
- The OCANNL `cuda_backend` pool-allocator rewrite — that is the downstream
  consumer, tracked as **task-6abfb6a9**, which this task blocks.
- The kernel-arg offset (handled in OCANNL codegen, no cudajit change).

**Dependencies**: none blocking. Relates to gh-ocannl-344 (the motivating pool
allocator) and blocks task-6abfb6a9 (the consumer rewrite).
