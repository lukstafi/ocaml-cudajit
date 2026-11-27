let kernel =
  {|
extern "C" __global__ void __launch_bounds__(64) shared_kernel(unsigned int *out) {
  // Volatile accesses preserve the full array through PTX generation and linking.
  __shared__ volatile unsigned int scratch[64];
  scratch[threadIdx.x] = threadIdx.x;
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
  let shared_bytes = Cu.Module.get_function_attribute func SHARED_SIZE_BYTES in
  let max_threads = Cu.Module.get_function_attribute func MAX_THREADS_PER_BLOCK in
  let valid_limit =
    max_threads > 0 && max_threads <= props.max_threads_per_block && max_threads <= 64
  in
  Printf.printf "static shared bytes = %d\n" shared_bytes;
  Printf.printf "thread limit respects device and kernel bounds: %b\n" valid_limit;
  ignore (Sys.opaque_identity context);
  if shared_bytes <> 256 || not valid_limit then exit 1
