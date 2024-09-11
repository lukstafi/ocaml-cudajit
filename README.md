# ocaml-cudajit

Bindings to the NVIDIA `cuda` and `nvrtc` libraries with a unified interface.

Requires a recent version of CUDA.

Paraphrased from the [SAXPY example](test/saxpy.ml):

```ocaml
let kernel =
  {| extern "C" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n) { ... } |}
module Cu = Cudajit
let prog = Cu.compile_to_ptx ~cu_src:kernel ~name:"saxpy" ~options:[ "--use_fast_math" ] ~with_debug:true
let () =
  Cu.init ();
  let device = Cu.device_get ~ordinal:0 in
  let context = Cu.ctx_create ~flags:0 device in
  let module_ = Cu.module_load_data_ex prog [] in
  let kernel = Cu.module_get_function module_ ~name:"saxpy" in
  ...
  let hX =
    Bigarray.Genarray.init Bigarray.Float32 Bigarray.C_layout [| size |] (fun idx -> Float.of_int idx.(0)) in
  let dX = Cu.alloc_and_memcpy hX in
  let dOut = Cu.alloc_and_memcpy hOut in
  Cu.launch_kernel kernel ~grid_dim_x:num_blocks ~block_dim_x:num_threads ~shared_mem_bytes:0 Cu.no_stream
    [
      Single a;
      Tensor dX;
      Tensor dY;
      Tensor dOut;
      Size_t Unsigned.Size_t.(mul (of_int num_threads) @@ of_int num_blocks);
    ];
  Cu.ctx_synchronize ();
  Cu.memcpy_D_to_H ~dst:hOut ~src:dOut ();
  Cu.mem_free dX;
  ...
  Cu.module_unload module_;
  Cu.ctx_destroy context
```

(the `...` are parts omitted for presentation brevity).
You can see how a kernel is compiled and launched, how on-device tensors are created, retrieved to host
(i.e. the CPU), and released.

Note that you don't need to add the include path to the `compile_to_ptx` options.

[Automatically generated API docs.](https://lukstafi.github.io/ocaml-cudajit/cudajit/index.html)
