# Add a non-owning "allocated pointer + offset" device-region view to cudajit

## Goal

OCANNL's Universal Pool Allocator (gh-ocannl-344) backs many tnodes with one
slab `Deviceptr.t` and addresses each sub-region at a distinct byte offset.
Sibling task-66a3bbff (merged as PR #9, `origin/main` `0bc8db8`) covered the
**copy** primitives by adding `?dst_offset` / `?src_offset` byte args to
`memcpy_H_to_D` / `memcpy_D_to_H` / `memcpy_D_to_D` (and their async `Stream`
variants). Wiring up the OCANNL CUDA pool allocator (task-6abfb6a9, which this
task **blocks**) surfaced three more sites where a bump-packed (non-zero
offset) sub-region must be addressed, and where the per-primitive `?offset`
approach does **not** reach:

- **kernel arguments** — `Stream.launch_kernel` consumes
  `kernel_param = Tensor of Deviceptr.t`; a kernel arg carries no offset at
  all, and threading an extra parallel-offsets array is the rejected option.
- **`memset`** (`Stream.memset_d8/d16/d32` + sync `Deviceptr.memset_d8/d16/d32`)
  — zero-initializing a non-zero-offset pool slot is currently impossible.
  ("memset should have offset args, that got overlooked." — user, 2026-06-23)
- **`memcpy_peer`** (sync `Deviceptr.memcpy_peer` + async `Stream.memcpy_peer`)
  — got no offsets in task-66a3bbff, so a pooled sub-region cannot cross
  devices. ("memcpy_peer should totally have offset args, that got
  overlooked." — user, 2026-06-23)

**Resolved direction (user, 2026-06-23):** introduce a non-owning "allocated
pointer + offset" *device-region view* type, rather than threading more
`?offset` int args site-by-site. This mirrors **tinygrad's CUDA backend**:
tinygrad passes every tensor as a separate kernel pointer arg
(`ops_cuda.py` `encode_args` — one `CUdeviceptr_v2` field per buffer) and forms
a sub-region by folding the byte offset **into the pointer value**
(`_offset(buf, _, off) = CUdeviceptr_v2(buf.value + off)`); the kernel still
dereferences from index 0. CUDA can do this because the device address is a raw
`uint64`. This is explicitly **not** Metal's slot-table / slab-base codegen
(opaque `MTLBuffer` + `(buffer, offset)` pairs); the CUDA backend stays on one
pointer per tensor.

## Acceptance Criteria

1. cudajit gains a **non-owning device-region view** type pairing a base
   `Deviceptr.t` with a byte offset. It is a **borrow**: it never owns or frees
   the allocation — only the base `Deviceptr.t` does — and it carries no
   finalizer. `Deviceptr.t` itself stays **abstract and owning**; no public raw
   `to_uint64`/`of_uint64` constructor or accessor is added.

2. The view is constructible from a `Deviceptr.t` and a byte offset, and a bare
   `Deviceptr.t` at any consuming site is addressable at offset 0 (whether via
   an explicit injection function or by the consuming site accepting both forms
   — the implementer's choice; see Approach).

3. The view is usable as a **kernel-launch argument**. When a kernel arg
   carries a non-zero offset, `launch_kernel` marshals `base + offset` as the
   `CUdeviceptr` for that arg (exactly tinygrad's `buf.value + off`); the kernel
   signature is **unchanged** (it still dereferences from index 0). The
   per-arg `check_freed` liveness guard and the existing
   `args_lifetimes` / `Remember` bookkeeping continue to apply to the base
   allocation.

4. The view is usable on the **`memset`** path: the synchronous
   `Deviceptr.memset_d8/d16/d32` and the asynchronous
   `Stream.memset_d8/d16/d32` can target a non-zero byte offset of an
   allocation, so a bump-packed pool slot can be zero-initialized.

5. The view is usable on the **`memcpy_peer`** path: both the synchronous
   `Deviceptr.memcpy_peer` and the asynchronous `Stream.memcpy_peer` accept a
   src and/or dst byte offset, so a pooled sub-region can cross devices.

6. **Backward compatibility:** every existing call site (kernel launch, memset,
   memcpy_peer) compiles and behaves identically — a bare `Deviceptr.t` at
   offset 0 is the unchanged path. The task-66a3bbff copy-primitive offsets
   (`?dst_offset`/`?src_offset` on `memcpy_H_to_D`/`memcpy_D_to_H`/
   `memcpy_D_to_D`) remain functional; whether they additionally accept the new
   view is the implementer's choice (see Scope) — they must not regress.

7. The `.mli` declares the new type and documents that it is a non-owning
   borrow (the base `Deviceptr` retains ownership; the view is never freed), the
   offset is in **bytes**, and offset 0 reproduces the prior behavior. Existing
   `cu*` reference doc links are retained on the touched functions. `CHANGES.md`
   gets an `### Added` entry under `[Unreleased]`.

8. Verified by a **runtest on minipc-wsl** that (a) launches a kernel against a
   non-zero-offset region of a larger allocation and confirms the kernel reads
   and/or writes the intended sub-region (neighbouring bytes untouched), and
   (b) exercises a non-zero-offset `memset` and reads it back to confirm the
   bytes land at the intended sub-region. The existing test suite continues to
   pass.

   AC verification reachability: the build/test host is `minipc-wsl` (reachable
   by passwordless SSH from mac-studio); the local `~/ocaml-cudajit` clone is a
   normal git worktree, so the implementation diff is verifiable by
   `git -C /Users/lukstafi/ocaml-cudajit diff`/`show` on the merge commit. The
   runtime AC (item 8) is GPU-bound and is verified by the test's PASS output on
   minipc-wsl, not by a SHA.

## Context

All paths are in `ocaml-cudajit/src/cuda.ml` / `cuda.mli` unless noted.
task-66a3bbff (`device-offset-memcpy.md`, merged `0bc8db8`) is the immediate
style precedent; this proposal extends the same idea to the three sites it left
out.

**The raw device pointer is a `uint64`, so an offset is plain arithmetic.**
`type deviceptr = Deviceptr of { ptr : memptr; freed : atomic_bool }` where
`type memptr = Unsigned.uint64`; the FFI `cu_deviceptr` is `uint64_t`. A device
byte offset is `Unsigned.UInt64.add ptr (Unsigned.UInt64.of_int offset)` applied
to the unpacked `ptr` field — no FFI/binding change is needed.

**The sibling already left a reusable helper.** task-66a3bbff added, at
top-level *before* `module Deviceptr` (so it is in scope for both the
`Deviceptr` and `Stream` modules):

```ocaml
let offset_deviceptr (Deviceptr { ptr; freed }) offset =
  if offset = 0 then Deviceptr { ptr; freed }
  else Deviceptr { ptr = Unsigned.UInt64.add ptr (Unsigned.UInt64.of_int offset); freed }
```

It "shares the original allocation's `freed` flag and is never returned to
callers nor finalized." This is exactly the **internal** mechanics of the
non-owning view; this task's job is to give that mechanics a **public,
named, type-level** form and route the three remaining sites through it.

**Kernel arguments** — `type kernel_param = Tensor of Deviceptr.t | Int … `
(in `module Stream`). `launch_kernel` marshals each `Tensor (Deviceptr { ptr;
freed })` by `check_freed` then `allocate uint64_t ptr` (the `base` address).
A non-zero offset folds in as `allocate uint64_t (base + offset)`. The
`args_lifetimes <- Remember (kernel_params, c_kernel_params) :: …` liveness
chain keeps the marshalled pointers alive across the async launch and must
continue to reference the base allocation's lifetime.

**`memset`** — sync `Deviceptr.memset_d8/d16/d32 (Deviceptr { ptr; freed }) v
~length` and async `Stream.memset_d8/d16/d32 (Deviceptr { ptr; freed }) v
~length stream`. Each `check_freed` then calls `Cuda.cu_memset_d*[_async] ptr
v (size_t length)`. The offset advances `ptr` before the call (via
`offset_deviceptr`), exactly as the copy primitives do. Note `length` for
`memset_d16`/`d32` is in *elements*, matching the existing `.mli` docs — the
offset added is in *bytes* (consistent with the copy primitives and the device
granularity).

**`memcpy_peer`** — sync `Deviceptr.memcpy_peer ?kind ?length ?size_in_bytes
~dst:(Deviceptr { ptr = dst; … }) ~dst_ctx ~src:(Deviceptr { ptr = src; … })
~src_ctx ()` and the async `Stream.memcpy_peer` (extra `stream` arg). Both
unpack both `ptr` fields and call `Cuda.cu_memcpy_peer[_async] dst dst_ctx src
src_ctx size`. Offsets advance `dst`/`src` before the call. This mirrors the
existing `memcpy_D_to_D` two-ended offset shape from task-66a3bbff.

**The Metal/CUDA-codegen consumer (downstream, not this repo):** the OCANNL
`cuda_backend` rewrite (task-6abfb6a9) is the consumer. Once this view exists,
`Slab.memset_zero device ~pool_id ~offset` in `allocate_delta` (`backends.ml`)
and the kernel-arg marshalling become the same `base + offset` pattern OCANNL's
shipped Metal backend already uses (offset applied at the operation, not folded
into a handle).

**Existing tests:** `test/test_memory.ml` exercises memset and copy with
deterministic patterns and PASS/FAIL prints (`test_memory_set_operations`,
`test_device_offset_transfers` from the sibling); the new offset-launch +
offset-memset tests should follow that style and register in `test/dune`. A
kernel-launch test pattern lives in `test/saxpy.ml`.

**tinygrad reference:** `~/tinygrad/tinygrad/runtime/ops_cuda.py` — `_offset`
(`buf.value + off`) and `encode_args` (one `CUdeviceptr_v2` per buffer).

## Approach

*Suggested approach — agents may deviate if they find a better path.* The API
shape (type name, record vs. abstract+accessor, whether `kernel_param.Tensor`
takes the view directly or a new `Tensor_at` constructor is added, whether the
copy primitives converge onto the view) is a **creative choice** the user
deliberately left open — this section sketches one coherent option, not a
mandate.

One coherent shape, building on the sibling's `offset_deviceptr`:

- A public type, e.g. `Deviceptr.region` (or a top-level `device_region`),
  carrying a base `Deviceptr.t` + byte offset, with a constructor
  `Deviceptr.offset : t -> bytes:int -> region` and an offset-0 injection so a
  bare `Deviceptr.t` is uniformly usable. Document it as a non-owning borrow.
- **Kernel params:** either change `Tensor of Deviceptr.t` to carry the view,
  or add a `Tensor_at of region` constructor (lower blast radius — the existing
  `Tensor` arm stays). In `launch_kernel`, resolve the view to `base + offset`
  and `allocate uint64_t` that; keep `check_freed` and `args_lifetimes` on the
  base.
- **memset / memcpy_peer:** route the unpacked `ptr` through the view's offset
  (reusing `offset_deviceptr` internally) before the `Cuda.cu_*` call. For
  `memset`, prefer accepting the view (or a `?offset` byte arg) so the bare-ptr
  call site is unchanged; for `memcpy_peer`, mirror `memcpy_D_to_D`'s
  `?dst_offset`/`?src_offset` or accept views for both ends.

Keep the diff local and mechanical at each site; the `freed` flag and
`check_freed` guards always stay on the base allocation, never on the borrow.

## Scope

**In scope:** a public non-owning device-region view type; routing
`Stream.launch_kernel` (kernel args), the sync + async `memset_d8/d16/d32`, and
the sync + async `memcpy_peer` through it; `.mli` type + docs; `CHANGES.md`
entry; one runtest on minipc-wsl covering offset kernel-launch and offset
memset.

**Out of scope / implementer's discretion:**
- Whether the task-66a3bbff copy primitives' existing `?dst_offset`/
  `?src_offset` args are *migrated* onto the new view or left as-is for backward
  compat (they are merged and already consumed). Either is acceptable provided
  AC 6 holds (no regression). Converging them is a nicety, not a requirement.
- Re-exposing a raw `Deviceptr.t` constructor or `to_uint64`/`of_uint64` —
  explicitly **not** wanted; the view is the only new surface.
- Metal-style `Pooled` codegen (slab base + slot table + kernel prologue) — the
  CUDA backend stays one pointer per tensor.
- The OCANNL `cuda_backend` pool-allocator rewrite — that is the downstream
  consumer, **task-6abfb6a9**, which this task blocks.

**Dependencies:** none blocking. Builds on task-66a3bbff (merged). Relates to
gh-ocannl-344 (the motivating pool allocator) and blocks task-6abfb6a9.

**Build/test host:** minipc-wsl only (RTX 3050 Ti, CUDA 12.8, cudajit on `main`
in the `5.4.0` switch; `git fetch` the local clone first). Same host + pattern
as task-66a3bbff.
