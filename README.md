# ocaml-cudajit

Bindings to the NVIDIA `cuda` and `nvrtc` libraries with a unified interface.

This project is sponsored by [Ahrefs](https://ocaml.org/success-stories/peta-byte-scale-web-crawler)! [Visit the Ahrefs website](https://ahrefs.com/).

Requires a recent version of CUDA.

Paraphrased from the [SAXPY example](test/saxpy.ml):

```ocaml
let kernel =
  {| extern "C" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n) { ... } |}
module Cu = Cuda
let prog =
  Nvrtc.compile_to_ptx ~cu_src:kernel ~name:"saxpy" ~options:[ "--use_fast_math" ]
    ~with_debug:true
let () =
  Cu.init ();
  let device = Cu.Device.get ~ordinal:0 in
  let context = Cu.Context.create ~flags:0 device in
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
  (* See if the finalizers run. *)
  Gc.full_major ()
```

You can see how a kernel is compiled and launched, how on-device tensors are created, retrieved to host
(i.e. the CPU), and released.

Note that you don't need to add the include path to the `compile_to_ptx` options.

[Automatically generated API docs.](https://lukstafi.github.io/ocaml-cudajit/cudajit/index.html)
