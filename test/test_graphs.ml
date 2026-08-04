(* Stream capture into a CUDA graph, instantiation, and repeated replay. The capture targets a
   non-default stream: the legacy NULL stream cannot be captured. *)

let kernel =
  {|
extern "C" __global__ void add_scalar(float *buf, float v, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    buf[tid] = buf[tid] + v;
  }
}
|}

let () =
  let num_blocks = 4 in
  let num_threads = 32 in
  let module Cu = Cuda in
  let prog = Nvrtc.compile_to_ptx ~cu_src:kernel ~name:"add_scalar" ~options:[] ~with_debug:true in
  Cu.init ();
  if Cu.Device.get_count () > 0 then (
    let device = Cu.Device.get ~ordinal:0 in
    let context = Cu.Context.create [] device in
    let module_ = Cu.Module.load_data_ex prog [] in
    let kernel = Cu.Module.get_function module_ ~name:"add_scalar" in
    let size = num_threads * num_blocks in
    let module Host = Bigarray.Genarray in
    let hBuf = Host.init Bigarray.Float32 Bigarray.C_layout [| size |] (fun _ -> 0.0) in
    let dBuf = Cu.Deviceptr.alloc_and_memcpy hBuf in
    let stream = Cu.Stream.create () in
    let launch v =
      Cu.Stream.launch_kernel kernel ~grid_dim_x:num_blocks ~block_dim_x:num_threads
        ~shared_mem_bytes:0 stream
        [ Tensor dBuf; Single v; Size_t (Unsigned.Size_t.of_int size) ]
    in
    (* Captured work is recorded, not executed: the buffer stays zero until the graph launches. *)
    Cu.Graph.begin_capture stream;
    launch 1.0;
    launch 2.0;
    let graph = Cu.Graph.end_capture stream in
    let exec = Cu.Graph.instantiate graph in
    Cu.Graph.destroy graph;
    Cu.Stream.synchronize stream;
    Cu.Deviceptr.memcpy_D_to_H ~dst:hBuf ~src:dBuf ();
    Printf.printf "after capture, before launch: buf[0] = %.1f\n" (Host.get hBuf [| 0 |]);
    (* Each replay adds 1.0 then 2.0 to every element. *)
    Cu.Graph.launch exec stream;
    Cu.Graph.launch exec stream;
    Cu.Stream.synchronize stream;
    Cu.Deviceptr.memcpy_D_to_H ~dst:hBuf ~src:dBuf ();
    Printf.printf "after two launches: buf[0] = %.1f, buf[%d] = %.1f\n" (Host.get hBuf [| 0 |])
      (size - 1)
      (Host.get hBuf [| size - 1 |]);
    Cu.Graph.exec_destroy exec;
    (* Idempotent: a second destroy and the GC finalizer are both no-ops. *)
    Cu.Graph.exec_destroy exec;
    ignore (Sys.opaque_identity context);
    Gc.full_major ();
    Printf.printf "done\n")
