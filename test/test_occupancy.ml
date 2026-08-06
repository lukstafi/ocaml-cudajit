(* Occupancy queries: how many blocks of a given size fit on a multiprocessor. The numbers are
   device-specific, so the test checks the invariants that hold on every device rather than the
   raw values. *)

let kernel =
  {|
extern "C" __global__ void scale(float *buf, float v, size_t n) {
  extern __shared__ float scratch[];
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    scratch[threadIdx.x] = buf[tid] * v;
    buf[tid] = scratch[threadIdx.x];
  }
}
|}

(* The printed booleans are compared against test_occupancy.expected by the (tests) stanza's
   automatic diff rule, but a promoted expected file would silently accept a regression, so every
   invariant is also counted here and a failure exits nonzero. *)
let failures = ref 0

let holds b =
  if not b then incr failures;
  b

let () =
  let module Cu = Cuda in
  Cu.init ();
  if Cu.Device.get_count () > 0 then (
    let device = Cu.Device.get ~ordinal:0 in
    let context = Cu.Context.create [] device in
    let props = Cu.Device.get_attributes device in
    let prog = Nvrtc.compile_to_ptx ~cu_src:kernel ~name:"scale" ~options:[] ~with_debug:false in
    let module_ = Cu.Module.load_data_ex prog [] in
    let func = Cu.Module.get_function module_ ~name:"scale" in
    let occupancy ?dynamic_smem_bytes block_size =
      Cu.Module.max_active_blocks_per_multiprocessor ?dynamic_smem_bytes func ~block_size
    in
    let block_sizes = [ 32; 64; 128; 256; 512; 1024 ] in
    let blocks = List.map (fun block_size -> (block_size, occupancy block_size)) block_sizes in
    List.iter
      (fun (block_size, n) ->
        (* A resource-light kernel fits at least one block of any legal size, and the resident
           threads cannot exceed what a multiprocessor can schedule. *)
        Printf.printf "block_size %4d: positive = %b, threads within limit = %b\n" block_size
          (holds (n > 0))
          (holds (n * block_size <= props.max_threads_per_multiprocessor)))
      blocks;
    (* Bigger blocks consume more of every per-multiprocessor resource, so the block count is
       non-increasing, and it never exceeds the device's hard per-multiprocessor block limit. *)
    let counts = List.map snd blocks in
    let rec non_increasing = function
      | a :: (b :: _ as rest) -> a >= b && non_increasing rest
      | _ -> true
    in
    Printf.printf "non-increasing in block size: %b\n" (holds (non_increasing counts));
    Printf.printf "within max_blocks_per_multiprocessor: %b\n"
      (holds (List.for_all (fun n -> n <= props.max_blocks_per_multiprocessor) counts));
    (* Dynamic shared memory competes with the blocks: asking for a sixteenth of a
       multiprocessor's shared memory per block caps residency at 16 blocks. *)
    let smem = props.max_shared_memory_per_multiprocessor / 16 in
    let with_smem = occupancy ~dynamic_smem_bytes:smem 128 in
    Printf.printf "shared memory limits residency: %b\n"
      (holds (with_smem <= 16 && with_smem <= occupancy 128));
    (* Configurations that cannot be launched at all report 0 rather than raising. *)
    Printf.printf "unlaunchable configurations report 0: %b\n"
      (holds
         (occupancy ~dynamic_smem_bytes:(2 * props.max_shared_memory_per_multiprocessor) 128 = 0
         && occupancy (2 * props.max_threads_per_block) = 0));
    (* Full-device grid dimension derived from the occupancy of the launch we actually make. *)
    let block_size = 128 in
    let grid_dim_x = occupancy block_size * props.multiprocessor_count in
    Printf.printf "grid fills the device: %b\n"
      (holds (grid_dim_x > 0 && grid_dim_x mod props.multiprocessor_count = 0));
    let module Host = Bigarray.Genarray in
    let size = grid_dim_x * block_size in
    let hBuf = Host.init Bigarray.Float32 Bigarray.C_layout [| size |] (fun _ -> 2.0) in
    let dBuf = Cu.Deviceptr.alloc_and_memcpy hBuf in
    Cu.Stream.launch_kernel func ~grid_dim_x ~block_dim_x:block_size
      ~shared_mem_bytes:(block_size * 4) Cu.Stream.no_stream
      [ Tensor dBuf; Single 3.0; Size_t (Unsigned.Size_t.of_int size) ];
    Cu.Context.synchronize ();
    Cu.Deviceptr.memcpy_D_to_H ~dst:hBuf ~src:dBuf ();
    Printf.printf "launch at that grid size: first = %.1f, last = %.1f\n" (Host.get hBuf [| 0 |])
      (Host.get hBuf [| size - 1 |]);
    (* 2.0 *. 3.0 is exact in binary32, so an equality test is safe here. *)
    ignore (holds (Host.get hBuf [| 0 |] = 6.0 && Host.get hBuf [| size - 1 |] = 6.0) : bool);
    ignore (Sys.opaque_identity context);
    Printf.printf "done\n";
    if !failures > 0 then exit 1)
