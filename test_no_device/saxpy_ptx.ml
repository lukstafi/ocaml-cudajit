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

let compilation () =
  let prog =
    Nvrtc.compile_to_ptx ~cu_src:kernel ~name:"saxpy" ~options:[ "--use_fast_math" ]
      ~with_debug:true
  in
  (match Nvrtc.compilation_log prog with
  | None -> ()
  | Some log -> Format.printf "\nCUDA Compile log: %s\n%!" log);
  Format.printf "PTX: %s%!"
  @@ Str.global_replace
       (Str.regexp
          {|CL-[0-9]+\|release [0-9]+\.[0-9]+\|V[0-9]+\.[0-9]+\.[0-9]+\|NVVM [0-9]+\.[0-9]+\.[0-9]+\|\.version [0-9]+\.[0-9]+\|\.target sm_[0-9]+|})
       "NNN"
  @@ Nvrtc.string_from_ptx prog

let kernel_half_prec =
  {|
    #include <cuda_fp16.h>
    extern "C" __global__ void saxpy(half a, half *x, half *y, half *out, size_t n) {
      size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) {
        // TODO: why doesn't this compile?
        // out[tid] = __hfma(a, x[tid], y[tid]);

        // This does not work as intended: it casts to fp32 instead of using the overloaded operators.
        // out[tid] = a * x[tid] + y[tid];

        out[tid] = __hadd(__hmul(a, x[tid]), y[tid]);
      }
    }
    |}

let half_precision_compilation () =
  let prog =
    Nvrtc.compile_to_ptx ~cu_src:kernel_half_prec ~name:"saxpy_half" ~options:[ "--use_fast_math" ]
      ~with_debug:true
  in
  (match Nvrtc.compilation_log prog with
  | None -> ()
  | Some log -> Format.printf "\nCUDA Compile log: %s\n%!" log);
  Format.printf "PTX: %s%!"
  @@ Str.global_replace
       (Str.regexp
          {|CL-[0-9]+\|release [0-9]+\.[0-9]+\|V[0-9]+\.[0-9]+\.[0-9]+\|NVVM [0-9]+\.[0-9]+\.[0-9]+\|\.version [0-9]+\.[0-9]+\|\.target sm_[0-9]+|})
       "NNN"
  @@ Nvrtc.string_from_ptx prog

let () =
  compilation ();
  half_precision_compilation ()
