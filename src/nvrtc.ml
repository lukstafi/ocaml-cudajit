open Nvrtc_ffi.Bindings_types
module Nvrtc_funs = Nvrtc_ffi.C.Functions
open Sexplib0.Sexp_conv

type result = nvrtc_result
(** See {{:https://docs.nvidia.com/cuda/nvrtc/index.html#_CPPv411nvrtcResult} enum nvrtcResult}. *)

let sexp_of_result = function
  | NVRTC_SUCCESS -> Sexplib0.Sexp.Atom "NVRTC_SUCCESS"
  | NVRTC_ERROR_OUT_OF_MEMORY -> Sexplib0.Sexp.Atom "NVRTC_ERROR_OUT_OF_MEMORY"
  | NVRTC_ERROR_PROGRAM_CREATION_FAILURE -> Sexplib0.Sexp.Atom "NVRTC_ERROR_PROGRAM_CREATION_FAILURE"
  | NVRTC_ERROR_INVALID_INPUT -> Sexplib0.Sexp.Atom "NVRTC_ERROR_INVALID_INPUT"
  | NVRTC_ERROR_INVALID_PROGRAM -> Sexplib0.Sexp.Atom "NVRTC_ERROR_INVALID_PROGRAM"
  | NVRTC_ERROR_INVALID_OPTION -> Sexplib0.Sexp.Atom "NVRTC_ERROR_INVALID_OPTION"
  | NVRTC_ERROR_COMPILATION -> Sexplib0.Sexp.Atom "NVRTC_ERROR_COMPILATION"
  | NVRTC_ERROR_BUILTIN_OPERATION_FAILURE -> Sexplib0.Sexp.Atom "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE"
  | NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION -> Sexplib0.Sexp.Atom "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION"
  | NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION -> Sexplib0.Sexp.Atom "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"
  | NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID -> Sexplib0.Sexp.Atom "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID"
  | NVRTC_ERROR_INTERNAL_ERROR -> Sexplib0.Sexp.Atom "NVRTC_ERROR_INTERNAL_ERROR"
  | NVRTC_ERROR_TIME_FILE_WRITE_FAILED -> Sexplib0.Sexp.Atom "NVRTC_ERROR_TIME_FILE_WRITE_FAILED"
  | NVRTC_ERROR_UNCATEGORIZED i -> Sexplib0.Sexp.List [Sexplib0.Sexp.Atom "NVRTC_ERROR_UNCATEGORIZED"; Sexplib0.Sexp.Atom (Int64.to_string i)]

exception Nvrtc_error of { status : result; message : string }

let error_printer = function
  | Nvrtc_error { status; message } ->
      ignore @@ Format.flush_str_formatter ();
      Format.fprintf Format.str_formatter "%s:@ %a" message Sexplib0.Sexp.pp_hum
        (sexp_of_result status);
      Some (Format.flush_str_formatter ())
  | _ -> None

let () = Printexc.register_printer error_printer
let is_success = function NVRTC_SUCCESS -> true | _ -> false

type compile_to_ptx_result = {
  log : string option;
  ptx : (char Ctypes.ptr[@sexp.opaque]);
  ptx_length : int;
}

let sexp_of_compile_to_ptx_result { log; ptx = _; ptx_length } =
  let log_sexp = match log with
    | None -> Sexplib0.Sexp.Atom "None"
    | Some s -> Sexplib0.Sexp.List [Sexplib0.Sexp.Atom "Some"; Sexplib0.Sexp.Atom s]
  in
  Sexplib0.Sexp.List [
    Sexplib0.Sexp.List [Sexplib0.Sexp.Atom "log"; log_sexp];
    Sexplib0.Sexp.List [Sexplib0.Sexp.Atom "ptx"; Sexplib0.Sexp.Atom "<opaque>"];
    Sexplib0.Sexp.List [Sexplib0.Sexp.Atom "ptx_length"; Sexplib0.Sexp.Atom (Int.to_string ptx_length)]
  ]

let compile_to_ptx ~cu_src ~name ~options ~with_debug =
  let open Ctypes in
  let prog = allocate_n nvrtc_program ~count:1 in
  (* We can add the include at the library level, because conf-cuda sets CUDA_PATH if it is missing
     but the information is available. *)
  let cuda_path =
    try Sys.getenv "CUDA_PATH"
    with Not_found -> (
      if not Sys.win32 then "/usr/local/cuda"
      else
        try Sys.getenv "LOCALAPPDATA" ^ "\\cuda_path_link"
        with Not_found -> "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8")
  in
  if not (Sys.file_exists cuda_path) then
    failwith (Printf.sprintf "CUDA_PATH %s does not exist" cuda_path);
  let options = Array.of_list @@ (("-I" ^ Filename.concat cuda_path "include") :: options) in
  let status =
    Nvrtc_funs.nvrtc_create_program prog cu_src name 0 (from_voidp string null)
      (from_voidp string null)
  in
  if status <> NVRTC_SUCCESS then
    raise @@ Nvrtc_error { status; message = "nvrtc_create_program " ^ name };
  let num_options = Array.length options in
  let get_c_options options =
    let c_options = CArray.make (ptr char) num_options in
    Array.iteri (fun i v -> CArray.of_string v |> CArray.start |> CArray.set c_options i) options;
    c_options
  in
  let c_options = get_c_options options in
  let valid_options =
    snd
    @@ CArray.fold_left
         (fun (i, valid) pchar ->
           if not valid then (i + 1, false)
           else
             let old_str = options.(i) in
             let str = Ctypes.string_from_ptr pchar ~length:(String.length old_str) in
             ( i + 1,
               String.for_all
                 (function
                   | 'a' .. 'z'
                   | 'A' .. 'Z'
                   | '0' .. '9'
                   | '-' | '_' | ':' | '/' | '\\' | ' ' | '"' | '.' | ';' | '&' | '#' | '%' | ',' ->
                       true
                   | _ -> false)
                 str ))
         (0, true) c_options
  in
  let default_options = [ "--use_fast_math"; "--device-debug" ] in
  let c_options =
    if valid_options then c_options
    else (
      Printf.printf "WARNING: Cudajit.Nvrtc.compile_to_ptx garbled options %s, using %s instead\n%!"
        (String.concat ", " @@ Array.to_list options)
        (String.concat ", " default_options);
      get_c_options @@ Array.of_list default_options)
  in
  let status = Nvrtc_funs.nvrtc_compile_program !@prog num_options @@ CArray.start c_options in
  let log_msg log = Option.value log ~default:"no compilation log" in
  let error prefix status log =
    ignore @@ Nvrtc_funs.nvrtc_destroy_program prog;
    raise @@ Nvrtc_error { status; message = prefix ^ " " ^ name ^ ": " ^ log_msg log }
  in
  let log =
    if status = NVRTC_SUCCESS && not with_debug then None
    else
      let log_size = allocate size_t Unsigned.Size_t.zero in
      let status = Nvrtc_funs.nvrtc_get_program_log_size !@prog log_size in
      if status <> NVRTC_SUCCESS then None
      else
        let count = Unsigned.Size_t.to_int !@log_size in
        let log = allocate_n char ~count in
        let status = Nvrtc_funs.nvrtc_get_program_log !@prog log in
        if status = NVRTC_SUCCESS then Some (string_from_ptr log ~length:(count - 1)) else None
  in
  if status <> NVRTC_SUCCESS then error "nvrtc_compile_program" status log;
  let ptx_size = allocate size_t Unsigned.Size_t.zero in
  let status = Nvrtc_funs.nvrtc_get_PTX_size !@prog ptx_size in
  if status <> NVRTC_SUCCESS then error "nvrtc_get_PTX_size" status log;
  let count = Unsigned.Size_t.to_int !@ptx_size in
  let ptx = allocate_n char ~count in
  let status = Nvrtc_funs.nvrtc_get_PTX !@prog ptx in
  if status <> NVRTC_SUCCESS then error "nvrtc_get_PTX" status log;
  ignore @@ Nvrtc_funs.nvrtc_destroy_program prog;
  { log; ptx; ptx_length = count - 1 }

let string_from_ptx prog = Ctypes.string_from_ptr prog.ptx ~length:prog.ptx_length
let compilation_log prog = prog.log
