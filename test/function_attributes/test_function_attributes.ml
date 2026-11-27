let kernel =
  {|
__constant__ unsigned int offsets[16];

extern "C" __global__ void __launch_bounds__(64) shared_kernel(unsigned int *out) {
  // Volatile accesses preserve the full array through PTX generation and linking.
  __shared__ volatile unsigned int scratch[64];
  scratch[threadIdx.x] = offsets[threadIdx.x % 16];
  __syncthreads();
  out[threadIdx.x] = scratch[threadIdx.x];
}
|}

let () =
  let module Cu = Cuda in
  Cu.init ();
  let device = Cu.Device.get ~ordinal:0 in
  let context = Cu.Context.create [] device in
  let props = Cu.Device.get_attributes device in
  let ptx =
    Nvrtc.compile_to_ptx ~cu_src:kernel ~name:"shared_kernel" ~options:[] ~with_debug:false
  in
  let module_ = Cu.Module.load_data_ex ptx [] in
  let func = Cu.Module.get_function module_ ~name:"shared_kernel" in
  let attribute = Cu.Module.get_function_attribute func in
  let shared_bytes = attribute SHARED_SIZE_BYTES in
  let max_threads = attribute MAX_THREADS_PER_BLOCK in
  let device_version = (10 * props.compute_capability_major) + props.compute_capability_minor in
  let checks =
    [
      ("thread limit is positive", max_threads > 0);
      ("thread limit respects the device limit", max_threads <= props.max_threads_per_block);
      ("thread limit respects the kernel launch bound", max_threads <= 64);
      ("register count is positive", attribute NUM_REGS > 0);
      ("constant memory covers offsets", attribute CONST_SIZE_BYTES >= 64);
      ("local memory is nonnegative", attribute LOCAL_SIZE_BYTES >= 0);
      ("PTX architecture version is positive", attribute PTX_VERSION > 0);
      ("JIT binary architecture matches the device", attribute BINARY_VERSION = device_version);
    ]
  in
  Printf.printf "static shared bytes = %d\n" shared_bytes;
  List.iter (fun (label, holds) -> Printf.printf "%s: %b\n" label holds) checks;
  ignore (Sys.opaque_identity context);
  (* A golden file promotion should still leave an invariant violation failing the test. *)
  if shared_bytes <> 256 || not (List.for_all snd checks) then exit 1
