(* The diagnostic check searches the error buffer up to its first NUL or capacity for the exact
   mnemonic injected by this test. *)
let ptx body : Nvrtc.compile_to_ptx_result =
  let source =
    ".version 7.0\n.target sm_50\n.address_size 64\n.visible .entry jit_probe() {\n" ^ body
    ^ "\n}\n"
  in
  let bytes = Ctypes.CArray.of_string source in
  { log = None; ptx = Ctypes.CArray.start bytes; ptx_length = String.length source }

let log_buffer () =
  let buffer = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4096 in
  Bigarray.Array1.fill buffer '\000';
  buffer

let log_mentions buffer marker =
  let bytes = Bytes.init (Bigarray.Array1.dim buffer) (Bigarray.Array1.get buffer) in
  let length = Option.value (Bytes.index_opt bytes '\000') ~default:(Bytes.length bytes) in
  let marker_length = String.length marker in
  let rec search offset =
    offset + marker_length <= length
    && (Bytes.sub_string bytes offset marker_length = marker || search (offset + 1))
  in
  search 0

let () =
  Cuda.init ();
  let context = Cuda.Context.create [] (Cuda.Device.get ~ordinal:0) in
  let image = ptx "ret;" in
  let load = Cuda.Module.load_data_ex in
  let get_probe module_ = ignore (Cuda.Module.get_function module_ ~name:"jit_probe") in
  get_probe (load image []);
  let info_log = log_buffer () in
  let error_log = log_buffer () in
  get_probe (load image [ ERROR_LOG_BUFFER error_log ]);
  get_probe
    (load image
       [
         ERROR_LOG_BUFFER error_log;
         INFO_LOG_BUFFER info_log;
         WALL_TIME { milliseconds = 0. };
         OPTIMIZATION_LEVEL 0;
         LOG_VERBOSE true;
       ]);
  (* Every load and symbol lookup above must succeed to reach this line. *)
  Printf.printf "valid PTX loads with error-only and paired log buffers: true\n";
  let bad_instruction = "cudajit_invalid_instruction" in
  let bad_image = ptx (bad_instruction ^ ";") in
  let error_log = log_buffer () in
  let diagnostic_names_instruction =
    match load bad_image [ ERROR_LOG_BUFFER error_log ] with
    | _ -> false
    | exception Cuda.Cuda_error { message = "cu_module_load_data_ex"; status } ->
        Cuda.sexp_of_result status = Sexplib0.Sexp.Atom "CUDA_ERROR_INVALID_PTX"
        && log_mentions error_log bad_instruction
  in
  Printf.printf "invalid PTX error log names the rejected instruction: %b\n"
    diagnostic_names_instruction;
  ignore (Sys.opaque_identity context);
  if not diagnostic_names_instruction then exit 1
