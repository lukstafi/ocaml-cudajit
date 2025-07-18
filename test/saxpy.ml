(* If the test fails, to verify your CUDA and NVRTC installation, follow the following instructions:
   https://docs.nvidia.com/cuda/nvrtc/index.html#code-saxpy-cpp and see where the OCaml version
   diverges. *)

let kernel =
  {|
extern "C" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = a * x[tid] + y[tid];
  }
}
|}

let () =
  let num_blocks = 32 in
  let num_threads = 128 in
  let module Cu = Cuda in
  Cu.cuda_call_hook := Some (fun ~message ~status:_ -> Printf.printf "%s\n" message);
  let prog =
    Nvrtc.compile_to_ptx ~cu_src:kernel ~name:"saxpy" ~options:[ "--use_fast_math" ]
      ~with_debug:true
  in
  Cu.init ();
  if Cu.Device.get_count () > 0 then (
    let device = Cu.Device.get ~ordinal:0 in
    let context = Cu.Context.create [] device in
    let module_ = Cu.Module.load_data_ex prog [] in
    let kernel = Cu.Module.get_function module_ ~name:"saxpy" in
    let size = num_threads * num_blocks in
    let module Host = Bigarray.Genarray in
    let a = 5.1 in
    let hX =
      Host.init Bigarray.Float32 Bigarray.C_layout [| size |] (fun idx -> Float.of_int idx.(0))
    in
    let dX = Cu.Deviceptr.alloc_and_memcpy hX in
    let hY =
      Host.init Bigarray.Float32 Bigarray.C_layout [| size |] (fun idx ->
          Float.of_int @@ (idx.(0) * 2))
    in
    let dY = Cu.Deviceptr.alloc_and_memcpy hY in
    let hOut = Host.create Bigarray.Float32 Bigarray.C_layout [| size |] in
    let dOut = Cu.Deviceptr.alloc_and_memcpy hOut in
    let eq = Cu.Deviceptr.equal in
    Printf.printf "dX = dX %b; dX = dY %b; dY = dOut %b.\n" (eq dX dX) (eq dX dY) (eq dY dOut);
    Cu.Stream.launch_kernel kernel ~grid_dim_x:num_blocks ~block_dim_x:num_threads
      ~shared_mem_bytes:0 Cu.Stream.no_stream
      [
        Single a;
        Tensor dX;
        Tensor dY;
        Tensor dOut;
        Size_t Unsigned.Size_t.(mul (of_int num_threads) @@ of_int num_blocks);
      ];
    Cu.Context.synchronize ();
    Cu.Deviceptr.memcpy_D_to_H ~dst:hOut ~src:dOut ();
    (* For testing reproducibility, synchronize garbage collection. *)
    ignore (Sys.opaque_identity context);
    Gc.full_major ();
    Format.set_margin 110;
    for i = 0 to size - 1 do
      let ( ! ) arr = Host.get arr [| i |] in
      Format.printf "%.1f * %.1f + %.1f = %.2f;@ " a !hX !hY !hOut
    done;
    Format.print_newline ());
  Cu.cuda_call_hook := None
