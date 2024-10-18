open Nvrtc_ffi.Bindings_types
module Nvrtc_funs = Nvrtc_ffi.C.Functions
module Cuda = Cuda_ffi.C.Functions
open Cuda_ffi.Bindings_types
open Sexplib0.Sexp_conv

module Nvrtc = struct
  type result = nvrtc_result [@@deriving sexp]
  (** See {{:https://docs.nvidia.com/cuda/nvrtc/index.html#_CPPv411nvrtcResult} enum nvrtcResult}. *)

  exception Nvrtc_error of { status : result; message : string }

  let error_printer = function
    | Nvrtc_error { status; message } ->
        ignore @@ Format.flush_str_formatter ();
        Format.fprintf Format.str_formatter "%s:@ %a" message Sexplib0.Sexp.pp_hum
          (sexp_of_result status);
        Some (Format.flush_str_formatter ())
    | _ -> None

  let () = Printexc.register_printer error_printer

  type compile_to_ptx_result = {
    log : string option;
    ptx : (char Ctypes.ptr[@sexp.opaque]);
    ptx_length : int;
  }
  [@@deriving sexp_of]

  let compile_to_ptx ~cu_src ~name ~options ~with_debug =
    let open Ctypes in
    let prog = allocate_n nvrtc_program ~count:1 in
    (* We can add the include at the library level, because conf-cuda sets CUDA_PATH if it is
       missing but the information is available. *)
    let default =
      if Sys.win32 then "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
      else "/usr/local/cuda"
    in
    let cuda_path = Sys.getenv_opt "CUDA_PATH" |> Option.value ~default in
    let options = ("-I" ^ Filename.concat cuda_path "include") :: options in
    let status =
      Nvrtc_funs.nvrtc_create_program prog cu_src name 0 (from_voidp string null)
        (from_voidp string null)
    in
    if status <> NVRTC_SUCCESS then
      raise @@ Nvrtc_error { status; message = "nvrtc_create_program " ^ name };
    let num_options = List.length options in
    let c_options = CArray.make (ptr char) num_options in

    List.iteri (fun i v -> CArray.of_string v |> CArray.start |> CArray.set c_options i) options;
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
end

type result = cu_result [@@deriving sexp]

exception Cuda_error of { status : result; message : string }

let cuda_error_printer = function
  | Cuda_error { status; message } ->
      ignore @@ Format.flush_str_formatter ();
      Format.fprintf Format.str_formatter "%s:@ %a" message Sexplib0.Sexp.pp_hum
        (sexp_of_result status);
      Some (Format.flush_str_formatter ())
  | _ -> None

let () = Printexc.register_printer cuda_error_printer
let check message status = if status <> CUDA_SUCCESS then raise @@ Cuda_error { status; message }
let init ?(flags = 0) () = check "cu_init" @@ Cuda.cu_init flags

type deviceptr = Deviceptr of Unsigned.uint64

let string_of_deviceptr (Deviceptr id) = Unsigned.UInt64.to_hexstring id
let sexp_of_deviceptr ptr = Sexplib0.Sexp.Atom (string_of_deviceptr ptr)

module Device = struct
  type t = cu_device [@@deriving sexp]

  let get_count () =
    let open Ctypes in
    let count = allocate int 0 in
    check "cu_device_get_count" @@ Cuda.cu_device_get_count count;
    !@count

  let get ~ordinal =
    let open Ctypes in
    let device = allocate Cuda_ffi.Types_generated.cu_device (Cu_device 0) in
    check "cu_device_get" @@ Cuda.cu_device_get device ordinal;
    !@device

  let primary_ctx_release device =
    check "cu_device_primary_ctx_release" @@ Cuda.cu_device_primary_ctx_release device

  let primary_ctx_reset device =
    check "cu_device_primary_ctx_reset" @@ Cuda.cu_device_primary_ctx_reset device

  let get_free_and_total_mem () =
    let open Ctypes in
    let free = allocate size_t Unsigned.Size_t.zero in
    let total = allocate size_t Unsigned.Size_t.zero in
    check "cu_mem_get_info" @@ Cuda.cu_mem_get_info free total;
    (Unsigned.Size_t.to_int !@free, Unsigned.Size_t.to_int !@total)

  type computemode = DEFAULT | PROHIBITED | EXCLUSIVE_PROCESS [@@deriving sexp]
  type flush_GPU_direct_RDMA_writes_options = HOST | MEMOPS [@@deriving sexp]

  type p2p_attribute =
    | PERFORMANCE_RANK of int
    | ACCESS_SUPPORTED of bool
    | NATIVE_ATOMIC_SUPPORTED of bool
    | CUDA_ARRAY_ACCESS_SUPPORTED of bool
  [@@deriving sexp]

  let get_p2p_attributes ~dst ~src =
    let open Ctypes in
    let result = ref [] in
    let value = allocate int 0 in
    check "cu_device_get_p2p_attribute"
    @@ Cuda.cu_device_get_p2p_attribute value CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK dst src;
    result := PERFORMANCE_RANK !@value :: !result;
    check "cu_device_get_p2p_attribute"
    @@ Cuda.cu_device_get_p2p_attribute value CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED dst src;
    result := ACCESS_SUPPORTED (!@value = 1) :: !result;
    check "cu_device_get_p2p_attribute"
    @@ Cuda.cu_device_get_p2p_attribute value CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED dst
         src;
    result := NATIVE_ATOMIC_SUPPORTED (!@value = 1) :: !result;
    check "cu_device_get_p2p_attribute"
    @@ Cuda.cu_device_get_p2p_attribute value CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED
         dst src;
    result := CUDA_ARRAY_ACCESS_SUPPORTED (!@value = 1) :: !result;
    !result

  let can_access_peer ~dst ~src =
    let open Ctypes in
    let can_access_peer = allocate int 0 in
    check "cu_device_can_access_peer" @@ Cuda.cu_device_can_access_peer can_access_peer dst src;
    !@can_access_peer <> 0

  let computemode_of_cu = function
    | CU_COMPUTEMODE_DEFAULT -> DEFAULT
    | CU_COMPUTEMODE_PROHIBITED -> PROHIBITED
    | CU_COMPUTEMODE_EXCLUSIVE_PROCESS -> EXCLUSIVE_PROCESS
    | CU_COMPUTEMODE_UNCATEGORIZED i -> invalid_arg @@ "Unknown computemode: " ^ Int64.to_string i

  let int_of_flush_GPU_direct_RDMA_writes_options =
    let open Cuda_ffi.Types_generated in
    function
    | HOST -> Int64.to_int cu_flush_gpu_direct_rdma_writes_option_host
    | MEMOPS -> Int64.to_int cu_flush_gpu_direct_rdma_writes_option_memops

  (* TODO: export CUmemAllocationHandleType to use in mempool_supported_handle_types. *)

  type mem_allocation_handle_type = NONE | POSIX_FILE_DESCRIPTOR | WIN32 | WIN32_KMT | FABRIC
  [@@deriving sexp]

  let int_of_mem_allocation_handle_type =
    let open Cuda_ffi.Types_generated in
    function
    | NONE -> Int64.to_int cu_mem_handle_type_none
    | POSIX_FILE_DESCRIPTOR -> Int64.to_int cu_mem_handle_type_posix_file_descriptor
    | WIN32 -> Int64.to_int cu_mem_handle_type_win32
    | WIN32_KMT -> Int64.to_int cu_mem_handle_type_win32_kmt
    | FABRIC -> Int64.to_int cu_mem_handle_type_fabric

  type attributes = {
    name : string;
    max_threads_per_block : int;
    max_block_dim_x : int;
    max_block_dim_y : int;
    max_block_dim_z : int;
    max_grid_dim_x : int;
    max_grid_dim_y : int;
    max_grid_dim_z : int;
    max_shared_memory_per_block : int;
    total_constant_memory : int;
    warp_size : int;
    max_pitch : int;
    max_registers_per_block : int;
    clock_rate : int;
    texture_alignment : int;
    multiprocessor_count : int;
    kernel_exec_timeout : bool;
    integrated : bool;
    can_map_host_memory : bool;
    compute_mode : computemode;
    maximum_texture1d_width : int;
    maximum_texture2d_width : int;
    maximum_texture2d_height : int;
    maximum_texture3d_width : int;
    maximum_texture3d_height : int;
    maximum_texture3d_depth : int;
    maximum_texture2d_layered_width : int;
    maximum_texture2d_layered_height : int;
    maximum_texture2d_layered_layers : int;
    surface_alignment : int;
    concurrent_kernels : bool;
    ecc_enabled : bool;
    pci_bus_id : int;
    pci_device_id : int;
    tcc_driver : bool;
    memory_clock_rate : int;
    global_memory_bus_width : int;
    l2_cache_size : int;
    max_threads_per_multiprocessor : int;
    async_engine_count : int;
    unified_addressing : bool;
    maximum_texture1d_layered_width : int;
    maximum_texture1d_layered_layers : int;
    maximum_texture2d_gather_width : int;
    maximum_texture2d_gather_height : int;
    maximum_texture3d_width_alternate : int;
    maximum_texture3d_height_alternate : int;
    maximum_texture3d_depth_alternate : int;
    pci_domain_id : int;
    texture_pitch_alignment : int;
    maximum_texturecubemap_width : int;
    maximum_texturecubemap_layered_width : int;
    maximum_texturecubemap_layered_layers : int;
    maximum_surface1d_width : int;
    maximum_surface2d_width : int;
    maximum_surface2d_height : int;
    maximum_surface3d_width : int;
    maximum_surface3d_height : int;
    maximum_surface3d_depth : int;
    maximum_surface1d_layered_width : int;
    maximum_surface1d_layered_layers : int;
    maximum_surface2d_layered_width : int;
    maximum_surface2d_layered_height : int;
    maximum_surface2d_layered_layers : int;
    maximum_surfacecubemap_width : int;
    maximum_surfacecubemap_layered_width : int;
    maximum_surfacecubemap_layered_layers : int;
    maximum_texture2d_linear_width : int;
    maximum_texture2d_linear_height : int;
    maximum_texture2d_linear_pitch : int;
    maximum_texture2d_mipmapped_width : int;
    maximum_texture2d_mipmapped_height : int;
    compute_capability_major : int;
    compute_capability_minor : int;
    maximum_texture1d_mipmapped_width : int;
    stream_priorities_supported : bool;
    global_l1_cache_supported : bool;
    local_l1_cache_supported : bool;
    max_shared_memory_per_multiprocessor : int;
    max_registers_per_multiprocessor : int;
    managed_memory : bool;
    multi_gpu_board : bool;
    multi_gpu_board_group_id : int;
    host_native_atomic_supported : bool;
    single_to_double_precision_perf_ratio : int;
    pageable_memory_access : bool;
    concurrent_managed_access : bool;
    compute_preemption_supported : bool;
    can_use_host_pointer_for_registered_mem : bool;
    cooperative_launch : bool;
    max_shared_memory_per_block_optin : int;
    can_flush_remote_writes : bool;
    host_register_supported : bool;
    pageable_memory_access_uses_host_page_tables : bool;
    direct_managed_mem_access_from_host : bool;
    virtual_memory_management_supported : bool;
    handle_type_posix_file_descriptor_supported : bool;
    handle_type_win32_handle_supported : bool;
    handle_type_win32_kmt_handle_supported : bool;
    max_blocks_per_multiprocessor : int;
    generic_compression_supported : bool;
    max_persisting_l2_cache_size : int;
    max_access_policy_window_size : int;
    gpu_direct_rdma_with_cuda_vmm_supported : bool;
    reserved_shared_memory_per_block : int;
    sparse_cuda_array_supported : bool;
    read_only_host_register_supported : bool;
    timeline_semaphore_interop_supported : bool;
    memory_pools_supported : bool;
    gpu_direct_rdma_supported : bool;
    gpu_direct_rdma_flush_writes_options : flush_GPU_direct_RDMA_writes_options list;
    gpu_direct_rdma_writes_ordering : bool;
    mempool_supported_handle_types : mem_allocation_handle_type list;
    cluster_launch : bool;
    deferred_mapping_cuda_array_supported : bool;
    can_use_64_bit_stream_mem_ops : bool;
    can_use_stream_wait_value_nor : bool;
    dma_buf_supported : bool;
    ipc_event_supported : bool;
    mem_sync_domain_count : int;
    tensor_map_access_supported : bool;
    unified_function_pointers : bool;
    multicast_supported : bool;
  }
  [@@deriving sexp]

  let get_attributes device =
    let open Ctypes in
    let count = 2048 in
    let name = allocate_n char ~count in
    check "cu_device_get_name" @@ Cuda.cu_device_get_name name count device;
    let name = coerce (ptr char) string name in
    let max_threads_per_block = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_threads_per_block CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
         device;
    let max_threads_per_block = !@max_threads_per_block in
    let max_block_dim_x = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_block_dim_x CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X device;
    let max_block_dim_x = !@max_block_dim_x in
    let max_block_dim_y = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_block_dim_y CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y device;
    let max_block_dim_y = !@max_block_dim_y in
    let max_block_dim_z = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_block_dim_z CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z device;
    let max_block_dim_z = !@max_block_dim_z in
    let max_grid_dim_x = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_grid_dim_x CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X device;
    let max_grid_dim_x = !@max_grid_dim_x in
    let max_grid_dim_y = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_grid_dim_y CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y device;
    let max_grid_dim_y = !@max_grid_dim_y in
    let max_grid_dim_z = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_grid_dim_z CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z device;
    let max_grid_dim_z = !@max_grid_dim_z in
    let max_shared_memory_per_block = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_shared_memory_per_block
         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK device;
    let max_shared_memory_per_block = !@max_shared_memory_per_block in
    let total_constant_memory = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute total_constant_memory CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
         device;
    let total_constant_memory = !@total_constant_memory in
    let warp_size = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute warp_size CU_DEVICE_ATTRIBUTE_WARP_SIZE device;
    let warp_size = !@warp_size in
    let max_pitch = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_pitch CU_DEVICE_ATTRIBUTE_MAX_PITCH device;
    let max_pitch = !@max_pitch in
    let max_registers_per_block = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_registers_per_block
         CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK device;
    let max_registers_per_block = !@max_registers_per_block in
    let clock_rate = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute clock_rate CU_DEVICE_ATTRIBUTE_CLOCK_RATE device;
    let clock_rate = !@clock_rate in
    let texture_alignment = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute texture_alignment CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT device;
    let texture_alignment = !@texture_alignment in
    let multiprocessor_count = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute multiprocessor_count CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
         device;
    let multiprocessor_count = !@multiprocessor_count in
    let kernel_exec_timeout = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute kernel_exec_timeout CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT
         device;
    let kernel_exec_timeout = 0 <> !@kernel_exec_timeout in
    let integrated = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute integrated CU_DEVICE_ATTRIBUTE_INTEGRATED device;
    let integrated = 0 <> !@integrated in
    let can_map_host_memory = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute can_map_host_memory CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
         device;
    let can_map_host_memory = 0 <> !@can_map_host_memory in
    let compute_mode = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute compute_mode CU_DEVICE_ATTRIBUTE_COMPUTE_MODE device;
    let compute_mode = computemode_of_cu @@ Cuda.cu_computemode_of_int !@compute_mode in
    let maximum_texture1d_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture1d_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH device;
    let maximum_texture1d_width = !@maximum_texture1d_width in
    let maximum_texture2d_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH device;
    let maximum_texture2d_width = !@maximum_texture2d_width in
    let maximum_texture2d_height = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_height
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT device;
    let maximum_texture2d_height = !@maximum_texture2d_height in
    let maximum_texture3d_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture3d_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH device;
    let maximum_texture3d_width = !@maximum_texture3d_width in
    let maximum_texture3d_height = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture3d_height
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT device;
    let maximum_texture3d_height = !@maximum_texture3d_height in
    let maximum_texture3d_depth = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture3d_depth
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH device;
    let maximum_texture3d_depth = !@maximum_texture3d_depth in
    let maximum_texture2d_layered_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_layered_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH device;
    let maximum_texture2d_layered_width = !@maximum_texture2d_layered_width in
    let maximum_texture2d_layered_height = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_layered_height
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT device;
    let maximum_texture2d_layered_height = !@maximum_texture2d_layered_height in
    let maximum_texture2d_layered_layers = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_layered_layers
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS device;
    let maximum_texture2d_layered_layers = !@maximum_texture2d_layered_layers in
    let surface_alignment = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute surface_alignment CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT device;
    let surface_alignment = !@surface_alignment in
    let concurrent_kernels = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute concurrent_kernels CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS device;
    let concurrent_kernels = 0 <> !@concurrent_kernels in
    let ecc_enabled = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute ecc_enabled CU_DEVICE_ATTRIBUTE_ECC_ENABLED device;
    let ecc_enabled = 0 <> !@ecc_enabled in
    let pci_bus_id = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute pci_bus_id CU_DEVICE_ATTRIBUTE_PCI_BUS_ID device;
    let pci_bus_id = !@pci_bus_id in
    let pci_device_id = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute pci_device_id CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID device;
    let pci_device_id = !@pci_device_id in
    let tcc_driver = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute tcc_driver CU_DEVICE_ATTRIBUTE_TCC_DRIVER device;
    let tcc_driver = 0 <> !@tcc_driver in
    let memory_clock_rate = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute memory_clock_rate CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE device;
    let memory_clock_rate = !@memory_clock_rate in
    let global_memory_bus_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute global_memory_bus_width
         CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH device;
    let global_memory_bus_width = !@global_memory_bus_width in
    let l2_cache_size = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute l2_cache_size CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE device;
    let l2_cache_size = !@l2_cache_size in
    let max_threads_per_multiprocessor = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_threads_per_multiprocessor
         CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR device;
    let max_threads_per_multiprocessor = !@max_threads_per_multiprocessor in
    let async_engine_count = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute async_engine_count CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT device;
    let async_engine_count = !@async_engine_count in
    let unified_addressing = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute unified_addressing CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING device;
    let unified_addressing = 0 <> !@unified_addressing in
    let maximum_texture1d_layered_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture1d_layered_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH device;
    let maximum_texture1d_layered_width = !@maximum_texture1d_layered_width in
    let maximum_texture1d_layered_layers = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture1d_layered_layers
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS device;
    let maximum_texture1d_layered_layers = !@maximum_texture1d_layered_layers in
    let maximum_texture2d_gather_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_gather_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH device;
    let maximum_texture2d_gather_width = !@maximum_texture2d_gather_width in
    let maximum_texture2d_gather_height = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_gather_height
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT device;
    let maximum_texture2d_gather_height = !@maximum_texture2d_gather_height in
    let maximum_texture3d_width_alternate = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture3d_width_alternate
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE device;
    let maximum_texture3d_width_alternate = !@maximum_texture3d_width_alternate in
    let maximum_texture3d_height_alternate = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture3d_height_alternate
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE device;
    let maximum_texture3d_height_alternate = !@maximum_texture3d_height_alternate in
    let maximum_texture3d_depth_alternate = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture3d_depth_alternate
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE device;
    let maximum_texture3d_depth_alternate = !@maximum_texture3d_depth_alternate in
    let pci_domain_id = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute pci_domain_id CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID device;
    let pci_domain_id = !@pci_domain_id in
    let texture_pitch_alignment = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute texture_pitch_alignment
         CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT device;
    let texture_pitch_alignment = !@texture_pitch_alignment in
    let maximum_texturecubemap_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texturecubemap_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH device;
    let maximum_texturecubemap_width = !@maximum_texturecubemap_width in
    let maximum_texturecubemap_layered_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texturecubemap_layered_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH device;
    let maximum_texturecubemap_layered_width = !@maximum_texturecubemap_layered_width in
    let maximum_texturecubemap_layered_layers = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texturecubemap_layered_layers
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS device;
    let maximum_texturecubemap_layered_layers = !@maximum_texturecubemap_layered_layers in
    let maximum_surface1d_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface1d_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH device;
    let maximum_surface1d_width = !@maximum_surface1d_width in
    let maximum_surface2d_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface2d_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH device;
    let maximum_surface2d_width = !@maximum_surface2d_width in
    let maximum_surface2d_height = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface2d_height
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT device;
    let maximum_surface2d_height = !@maximum_surface2d_height in
    let maximum_surface3d_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface3d_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH device;
    let maximum_surface3d_width = !@maximum_surface3d_width in
    let maximum_surface3d_height = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface3d_height
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT device;
    let maximum_surface3d_height = !@maximum_surface3d_height in
    let maximum_surface3d_depth = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface3d_depth
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH device;
    let maximum_surface3d_depth = !@maximum_surface3d_depth in
    let maximum_surface1d_layered_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface1d_layered_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH device;
    let maximum_surface1d_layered_width = !@maximum_surface1d_layered_width in
    let maximum_surface1d_layered_layers = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface1d_layered_layers
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS device;
    let maximum_surface1d_layered_layers = !@maximum_surface1d_layered_layers in
    let maximum_surface2d_layered_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface2d_layered_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH device;
    let maximum_surface2d_layered_width = !@maximum_surface2d_layered_width in
    let maximum_surface2d_layered_height = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface2d_layered_height
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT device;
    let maximum_surface2d_layered_height = !@maximum_surface2d_layered_height in
    let maximum_surface2d_layered_layers = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surface2d_layered_layers
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS device;
    let maximum_surface2d_layered_layers = !@maximum_surface2d_layered_layers in
    let maximum_surfacecubemap_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surfacecubemap_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH device;
    let maximum_surfacecubemap_width = !@maximum_surfacecubemap_width in
    let maximum_surfacecubemap_layered_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surfacecubemap_layered_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH device;
    let maximum_surfacecubemap_layered_width = !@maximum_surfacecubemap_layered_width in
    let maximum_surfacecubemap_layered_layers = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_surfacecubemap_layered_layers
         CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS device;
    let maximum_surfacecubemap_layered_layers = !@maximum_surfacecubemap_layered_layers in
    let maximum_texture2d_linear_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_linear_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH device;
    let maximum_texture2d_linear_width = !@maximum_texture2d_linear_width in
    let maximum_texture2d_linear_height = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_linear_height
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT device;
    let maximum_texture2d_linear_height = !@maximum_texture2d_linear_height in
    let maximum_texture2d_linear_pitch = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_linear_pitch
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH device;
    let maximum_texture2d_linear_pitch = !@maximum_texture2d_linear_pitch in
    let maximum_texture2d_mipmapped_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_mipmapped_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH device;
    let maximum_texture2d_mipmapped_width = !@maximum_texture2d_mipmapped_width in
    let maximum_texture2d_mipmapped_height = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture2d_mipmapped_height
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT device;
    let maximum_texture2d_mipmapped_height = !@maximum_texture2d_mipmapped_height in
    let compute_capability_major = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute compute_capability_major
         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR device;
    let compute_capability_major = !@compute_capability_major in
    let compute_capability_minor = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute compute_capability_minor
         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR device;
    let compute_capability_minor = !@compute_capability_minor in
    let maximum_texture1d_mipmapped_width = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute maximum_texture1d_mipmapped_width
         CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH device;
    let maximum_texture1d_mipmapped_width = !@maximum_texture1d_mipmapped_width in
    let stream_priorities_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute stream_priorities_supported
         CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED device;
    let stream_priorities_supported = 0 <> !@stream_priorities_supported in
    let global_l1_cache_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute global_l1_cache_supported
         CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED device;
    let global_l1_cache_supported = 0 <> !@global_l1_cache_supported in
    let local_l1_cache_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute local_l1_cache_supported
         CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED device;
    let local_l1_cache_supported = 0 <> !@local_l1_cache_supported in
    let max_shared_memory_per_multiprocessor = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_shared_memory_per_multiprocessor
         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR device;
    let max_shared_memory_per_multiprocessor = !@max_shared_memory_per_multiprocessor in
    let max_registers_per_multiprocessor = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_registers_per_multiprocessor
         CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR device;
    let max_registers_per_multiprocessor = !@max_registers_per_multiprocessor in
    let managed_memory = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute managed_memory CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY device;
    let managed_memory = 0 <> !@managed_memory in
    let multi_gpu_board = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute multi_gpu_board CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD device;
    let multi_gpu_board = 0 <> !@multi_gpu_board in
    let multi_gpu_board_group_id = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute multi_gpu_board_group_id
         CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID device;
    let multi_gpu_board_group_id = !@multi_gpu_board_group_id in
    let host_native_atomic_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute host_native_atomic_supported
         CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED device;
    let host_native_atomic_supported = 0 <> !@host_native_atomic_supported in
    let single_to_double_precision_perf_ratio = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute single_to_double_precision_perf_ratio
         CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO device;
    let single_to_double_precision_perf_ratio = !@single_to_double_precision_perf_ratio in
    let pageable_memory_access = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute pageable_memory_access
         CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS device;
    let pageable_memory_access = 0 <> !@pageable_memory_access in
    let concurrent_managed_access = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute concurrent_managed_access
         CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS device;
    let concurrent_managed_access = 0 <> !@concurrent_managed_access in
    let compute_preemption_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute compute_preemption_supported
         CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED device;
    let compute_preemption_supported = 0 <> !@compute_preemption_supported in
    let can_use_host_pointer_for_registered_mem = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute can_use_host_pointer_for_registered_mem
         CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM device;
    let can_use_host_pointer_for_registered_mem = 0 <> !@can_use_host_pointer_for_registered_mem in
    let cooperative_launch = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute cooperative_launch CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH device;
    let cooperative_launch = 0 <> !@cooperative_launch in
    let max_shared_memory_per_block_optin = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_shared_memory_per_block_optin
         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN device;
    let max_shared_memory_per_block_optin = !@max_shared_memory_per_block_optin in
    let can_flush_remote_writes = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute can_flush_remote_writes
         CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES device;
    let can_flush_remote_writes = 0 <> !@can_flush_remote_writes in
    let host_register_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute host_register_supported
         CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED device;
    let host_register_supported = 0 <> !@host_register_supported in
    let pageable_memory_access_uses_host_page_tables = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute pageable_memory_access_uses_host_page_tables
         CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES device;
    let pageable_memory_access_uses_host_page_tables =
      0 <> !@pageable_memory_access_uses_host_page_tables
    in
    let direct_managed_mem_access_from_host = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute direct_managed_mem_access_from_host
         CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST device;
    let direct_managed_mem_access_from_host = 0 <> !@direct_managed_mem_access_from_host in
    let virtual_memory_management_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute virtual_memory_management_supported
         CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED device;
    let virtual_memory_management_supported = 0 <> !@virtual_memory_management_supported in
    let handle_type_posix_file_descriptor_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute handle_type_posix_file_descriptor_supported
         CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED device;
    let handle_type_posix_file_descriptor_supported =
      0 <> !@handle_type_posix_file_descriptor_supported
    in
    let handle_type_win32_handle_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute handle_type_win32_handle_supported
         CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED device;
    let handle_type_win32_handle_supported = 0 <> !@handle_type_win32_handle_supported in
    let handle_type_win32_kmt_handle_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute handle_type_win32_kmt_handle_supported
         CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED device;
    let handle_type_win32_kmt_handle_supported = 0 <> !@handle_type_win32_kmt_handle_supported in
    let max_blocks_per_multiprocessor = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_blocks_per_multiprocessor
         CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR device;
    let max_blocks_per_multiprocessor = !@max_blocks_per_multiprocessor in
    let generic_compression_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute generic_compression_supported
         CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED device;
    let generic_compression_supported = 0 <> !@generic_compression_supported in
    let max_persisting_l2_cache_size = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_persisting_l2_cache_size
         CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE device;
    let max_persisting_l2_cache_size = !@max_persisting_l2_cache_size in
    let max_access_policy_window_size = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute max_access_policy_window_size
         CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE device;
    let max_access_policy_window_size = !@max_access_policy_window_size in
    let gpu_direct_rdma_with_cuda_vmm_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute gpu_direct_rdma_with_cuda_vmm_supported
         CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED device;
    let gpu_direct_rdma_with_cuda_vmm_supported = 0 <> !@gpu_direct_rdma_with_cuda_vmm_supported in
    let reserved_shared_memory_per_block = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute reserved_shared_memory_per_block
         CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK device;
    let reserved_shared_memory_per_block = !@reserved_shared_memory_per_block in
    let sparse_cuda_array_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute sparse_cuda_array_supported
         CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED device;
    let sparse_cuda_array_supported = 0 <> !@sparse_cuda_array_supported in
    let read_only_host_register_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute read_only_host_register_supported
         CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED device;
    let read_only_host_register_supported = 0 <> !@read_only_host_register_supported in
    let timeline_semaphore_interop_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute timeline_semaphore_interop_supported
         CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED device;
    let timeline_semaphore_interop_supported = 0 <> !@timeline_semaphore_interop_supported in
    let memory_pools_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute memory_pools_supported
         CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED device;
    let memory_pools_supported = 0 <> !@memory_pools_supported in
    let gpu_direct_rdma_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute gpu_direct_rdma_supported
         CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED device;
    let gpu_direct_rdma_supported = 0 <> !@gpu_direct_rdma_supported in
    let rec unfold f flags remaining =
      let open Int in
      match remaining with
      | [] ->
          if not (equal flags zero) then
            failwith @@ "ctx_get_flags: unknown flag " ^ to_string flags
          else []
      | flag :: remaining ->
          if equal flags zero then []
          else
            let uflag = f flag in
            if equal (flags land uflag) zero then unfold f flags remaining
            else flag :: unfold f (flags lxor uflag) remaining
    in
    let gpu_direct_rdma_flush_writes_options = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute gpu_direct_rdma_flush_writes_options
         CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS device;
    let gpu_direct_rdma_flush_writes_options =
      unfold int_of_flush_GPU_direct_RDMA_writes_options
        !@gpu_direct_rdma_flush_writes_options
        [ HOST; MEMOPS ]
    in
    let gpu_direct_rdma_writes_ordering = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute gpu_direct_rdma_writes_ordering
         CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING device;
    let gpu_direct_rdma_writes_ordering = 0 <> !@gpu_direct_rdma_writes_ordering in
    let mempool_supported_handle_types = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute mempool_supported_handle_types
         CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES device;
    let mempool_supported_handle_types =
      unfold int_of_mem_allocation_handle_type !@mempool_supported_handle_types
        [ NONE; POSIX_FILE_DESCRIPTOR; WIN32; WIN32_KMT; FABRIC ]
    in

    let cluster_launch = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute cluster_launch CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH device;
    let cluster_launch = 0 <> !@cluster_launch in
    let deferred_mapping_cuda_array_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute deferred_mapping_cuda_array_supported
         CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED device;
    let deferred_mapping_cuda_array_supported = 0 <> !@deferred_mapping_cuda_array_supported in
    let can_use_64_bit_stream_mem_ops = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute can_use_64_bit_stream_mem_ops
         CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS device;
    let can_use_64_bit_stream_mem_ops = 0 <> !@can_use_64_bit_stream_mem_ops in
    let can_use_stream_wait_value_nor = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute can_use_stream_wait_value_nor
         CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR device;
    let can_use_stream_wait_value_nor = 0 <> !@can_use_stream_wait_value_nor in
    let dma_buf_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute dma_buf_supported CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED device;
    let dma_buf_supported = 0 <> !@dma_buf_supported in
    let ipc_event_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute ipc_event_supported CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED
         device;
    let ipc_event_supported = 0 <> !@ipc_event_supported in
    let mem_sync_domain_count = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute mem_sync_domain_count CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT
         device;
    let mem_sync_domain_count = !@mem_sync_domain_count in
    let tensor_map_access_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute tensor_map_access_supported
         CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED device;
    let tensor_map_access_supported = 0 <> !@tensor_map_access_supported in
    let unified_function_pointers = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute unified_function_pointers
         CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS device;
    let unified_function_pointers = 0 <> !@unified_function_pointers in
    let multicast_supported = allocate int 0 in
    check "cu_device_get_attribute"
    @@ Cuda.cu_device_get_attribute multicast_supported CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
         device;
    let multicast_supported = 0 <> !@multicast_supported in
    {
      name;
      max_threads_per_block;
      max_block_dim_x;
      max_block_dim_y;
      max_block_dim_z;
      max_grid_dim_x;
      max_grid_dim_y;
      max_grid_dim_z;
      max_shared_memory_per_block;
      total_constant_memory;
      warp_size;
      max_pitch;
      max_registers_per_block;
      clock_rate;
      texture_alignment;
      multiprocessor_count;
      kernel_exec_timeout;
      integrated;
      can_map_host_memory;
      compute_mode;
      maximum_texture1d_width;
      maximum_texture2d_width;
      maximum_texture2d_height;
      maximum_texture3d_width;
      maximum_texture3d_height;
      maximum_texture3d_depth;
      maximum_texture2d_layered_width;
      maximum_texture2d_layered_height;
      maximum_texture2d_layered_layers;
      surface_alignment;
      concurrent_kernels;
      ecc_enabled;
      pci_bus_id;
      pci_device_id;
      tcc_driver;
      memory_clock_rate;
      global_memory_bus_width;
      l2_cache_size;
      max_threads_per_multiprocessor;
      async_engine_count;
      unified_addressing;
      maximum_texture1d_layered_width;
      maximum_texture1d_layered_layers;
      maximum_texture2d_gather_width;
      maximum_texture2d_gather_height;
      maximum_texture3d_width_alternate;
      maximum_texture3d_height_alternate;
      maximum_texture3d_depth_alternate;
      pci_domain_id;
      texture_pitch_alignment;
      maximum_texturecubemap_width;
      maximum_texturecubemap_layered_width;
      maximum_texturecubemap_layered_layers;
      maximum_surface1d_width;
      maximum_surface2d_width;
      maximum_surface2d_height;
      maximum_surface3d_width;
      maximum_surface3d_height;
      maximum_surface3d_depth;
      maximum_surface1d_layered_width;
      maximum_surface1d_layered_layers;
      maximum_surface2d_layered_width;
      maximum_surface2d_layered_height;
      maximum_surface2d_layered_layers;
      maximum_surfacecubemap_width;
      maximum_surfacecubemap_layered_width;
      maximum_surfacecubemap_layered_layers;
      maximum_texture2d_linear_width;
      maximum_texture2d_linear_height;
      maximum_texture2d_linear_pitch;
      maximum_texture2d_mipmapped_width;
      maximum_texture2d_mipmapped_height;
      compute_capability_major;
      compute_capability_minor;
      maximum_texture1d_mipmapped_width;
      stream_priorities_supported;
      global_l1_cache_supported;
      local_l1_cache_supported;
      max_shared_memory_per_multiprocessor;
      max_registers_per_multiprocessor;
      managed_memory;
      multi_gpu_board;
      multi_gpu_board_group_id;
      host_native_atomic_supported;
      single_to_double_precision_perf_ratio;
      pageable_memory_access;
      concurrent_managed_access;
      compute_preemption_supported;
      can_use_host_pointer_for_registered_mem;
      cooperative_launch;
      max_shared_memory_per_block_optin;
      can_flush_remote_writes;
      host_register_supported;
      pageable_memory_access_uses_host_page_tables;
      direct_managed_mem_access_from_host;
      virtual_memory_management_supported;
      handle_type_posix_file_descriptor_supported;
      handle_type_win32_handle_supported;
      handle_type_win32_kmt_handle_supported;
      max_blocks_per_multiprocessor;
      generic_compression_supported;
      max_persisting_l2_cache_size;
      max_access_policy_window_size;
      gpu_direct_rdma_with_cuda_vmm_supported;
      reserved_shared_memory_per_block;
      sparse_cuda_array_supported;
      read_only_host_register_supported;
      timeline_semaphore_interop_supported;
      memory_pools_supported;
      gpu_direct_rdma_supported;
      gpu_direct_rdma_flush_writes_options;
      gpu_direct_rdma_writes_ordering;
      mempool_supported_handle_types;
      cluster_launch;
      deferred_mapping_cuda_array_supported;
      can_use_64_bit_stream_mem_ops;
      can_use_stream_wait_value_nor;
      dma_buf_supported;
      ipc_event_supported;
      mem_sync_domain_count;
      tensor_map_access_supported;
      unified_function_pointers;
      multicast_supported;
    }
end

type bigstring = (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
type lifetime = Remember : 'a -> lifetime
type delimited_event = { event : cu_event; mutable is_released : bool }

let destroy_event event = check "cu_event_destroy" @@ Cuda.cu_event_destroy event

type stream = {
  mutable args_lifetimes : lifetime list;
  mutable owned_events : delimited_event list;
  stream : cu_stream;
}

let release_stream stream =
  stream.args_lifetimes <- [];
  List.iter
    (fun event ->
      if not event.is_released then destroy_event event.event;
      event.is_released <- true)
    stream.owned_events;
  stream.owned_events <- []

let no_stream =
  { args_lifetimes = []; owned_events = []; stream = Ctypes.(coerce (ptr void) cu_stream null) }

let sexp_of_voidp ptr =
  Sexplib0.Sexp.Atom
    ("@" ^ Unsigned.UInt64.to_hexstring @@ Unsigned.UInt64.of_string @@ Nativeint.to_string
   @@ Ctypes.raw_address_of_ptr ptr)

module Context = struct
  type t = cu_context

  let sexp_of_t (ctx : t) = sexp_of_voidp @@ Ctypes.to_voidp ctx

  type flag =
    | SCHED_AUTO
    | SCHED_SPIN
    | SCHED_YIELD
    | SCHED_BLOCKING_SYNC
    | SCHED_MASK
    | MAP_HOST
    | LMEM_RESIZE_TO_MAX
    | COREDUMP_ENABLE
    | USER_COREDUMP_ENABLE
    | SYNC_MEMOPS
  [@@deriving sexp]

  type flags = flag list [@@deriving sexp]

  let uint_of_flag f =
    let open Cuda_ffi.Types_generated in
    match f with
    | SCHED_AUTO -> Unsigned.UInt.of_int64 cu_ctx_sched_auto
    | SCHED_SPIN -> Unsigned.UInt.of_int64 cu_ctx_sched_spin
    | SCHED_YIELD -> Unsigned.UInt.of_int64 cu_ctx_sched_yield
    | SCHED_BLOCKING_SYNC -> Unsigned.UInt.of_int64 cu_ctx_sched_blocking_sync
    | SCHED_MASK -> Unsigned.UInt.of_int64 cu_ctx_sched_mask
    | MAP_HOST -> Unsigned.UInt.of_int64 cu_ctx_map_host
    | LMEM_RESIZE_TO_MAX -> Unsigned.UInt.of_int64 cu_ctx_lmem_resize_to_max
    | COREDUMP_ENABLE -> Unsigned.UInt.of_int64 cu_ctx_coredump_enable
    | USER_COREDUMP_ENABLE -> Unsigned.UInt.of_int64 cu_ctx_user_coredump_enable
    | SYNC_MEMOPS -> Unsigned.UInt.of_int64 cu_ctx_sync_memops

  let destroy ctx = check "cu_ctx_destroy" @@ Cuda.cu_ctx_destroy ctx

  let create (flags : flags) device =
    let open Ctypes in
    let ctx = allocate_n cu_context ~count:1 in
    let open Unsigned.UInt in
    let flags = List.fold_left (fun flags flag -> Infix.(flags lor uint_of_flag flag)) zero flags in
    check "cu_ctx_create" @@ Cuda.cu_ctx_create ctx flags device;
    let ctx = !@ctx in
    Stdlib.Gc.finalise destroy ctx;
    ctx

  let get_flags () : flags =
    let open Ctypes in
    let open Unsigned.UInt in
    let flags = allocate uint zero in
    check "cu_ctx_create" @@ Cuda.cu_ctx_get_flags flags;
    let mask = Unsigned.UInt.of_int64 Cuda_ffi.Types_generated.cu_ctx_flags_mask in
    let rec unfold flags remaining =
      match remaining with
      | [] ->
          if not (equal flags zero) then
            failwith @@ "ctx_get_flags: unknown flag " ^ to_string flags
          else []
      | flag :: remaining ->
          if equal flags zero then []
          else
            let uflag = uint_of_flag flag in
            if equal Infix.(flags land uflag) zero then unfold flags remaining
            else flag :: unfold Infix.(flags lxor uflag) remaining
    in
    unfold
      Infix.(!@flags land mask)
      [
        SCHED_AUTO;
        SCHED_SPIN;
        SCHED_YIELD;
        SCHED_BLOCKING_SYNC;
        SCHED_MASK;
        MAP_HOST;
        LMEM_RESIZE_TO_MAX;
        COREDUMP_ENABLE;
        USER_COREDUMP_ENABLE;
        SYNC_MEMOPS;
      ]

  let get_device () =
    let open Ctypes in
    let device = allocate Cuda_ffi.Types_generated.cu_device (Cu_device 0) in
    check "cu_ctx_get_device" @@ Cuda.cu_ctx_get_device device;
    !@device

  let pop_current () =
    let open Ctypes in
    let ctx = allocate_n cu_context ~count:1 in
    check "cu_ctx_pop_current" @@ Cuda.cu_ctx_pop_current ctx;
    !@ctx

  let get_current () =
    let open Ctypes in
    let ctx = allocate_n cu_context ~count:1 in
    check "cu_ctx_get_current" @@ Cuda.cu_ctx_get_current ctx;
    !@ctx

  let push_current ctx = check "cu_ctx_push_current" @@ Cuda.cu_ctx_push_current ctx
  let set_current ctx = check "cu_ctx_set_current" @@ Cuda.cu_ctx_set_current ctx

  let get_primary device =
    let open Ctypes in
    let ctx = allocate_n cu_context ~count:1 in
    check "cu_device_primary_ctx_retain" @@ Cuda.cu_device_primary_ctx_retain ctx device;
    let ctx = !@ctx in
    Stdlib.Gc.finalise (fun _ -> Device.primary_ctx_release device) ctx;
    ctx

  let synchronize () =
    check "cu_ctx_synchronize" @@ Cuda.cu_ctx_synchronize ();
    release_stream no_stream

  let disable_peer_access ctx =
    check "cu_ctx_disable_peer_access" @@ Cuda.cu_ctx_disable_peer_access ctx

  let enable_peer_access ?(flags = Unsigned.UInt.zero) ctx =
    check "cu_ctx_enable_peer_access" @@ Cuda.cu_ctx_enable_peer_access ctx flags

  type limit =
    | STACK_SIZE
    | PRINTF_FIFO_SIZE
    | MALLOC_HEAP_SIZE
    | DEV_RUNTIME_SYNC_DEPTH
    | DEV_RUNTIME_PENDING_LAUNCH_COUNT
    | MAX_L2_FETCH_GRANULARITY
    | PERSISTING_L2_CACHE_SIZE
  [@@deriving sexp]

  let cu_of_limit = function
    | STACK_SIZE -> CU_LIMIT_STACK_SIZE
    | PRINTF_FIFO_SIZE -> CU_LIMIT_PRINTF_FIFO_SIZE
    | MALLOC_HEAP_SIZE -> CU_LIMIT_MALLOC_HEAP_SIZE
    | DEV_RUNTIME_SYNC_DEPTH -> CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
    | DEV_RUNTIME_PENDING_LAUNCH_COUNT -> CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT
    | MAX_L2_FETCH_GRANULARITY -> CU_LIMIT_MAX_L2_FETCH_GRANULARITY
    | PERSISTING_L2_CACHE_SIZE -> CU_LIMIT_PERSISTING_L2_CACHE_SIZE

  let set_limit limit value =
    check "cu_ctx_set_limit"
    @@ Cuda.cu_ctx_set_limit (cu_of_limit limit)
    @@ Unsigned.Size_t.of_int value

  let get_limit limit =
    let open Ctypes in
    let value = allocate size_t Unsigned.Size_t.zero in
    check "cu_ctx_set_limit" @@ Cuda.cu_ctx_get_limit value (cu_of_limit limit);
    Unsigned.Size_t.to_int !@value
end

let bigarray_start_not_managed arr = Ctypes_bigarray.unsafe_address arr

let get_ptr_not_managed ~reftyp arr =
  (* Work around because Ctypes.bigarray_start doesn't support half precision. *)
  Ctypes_static.CPointer (Ctypes_memory.make_unmanaged ~reftyp @@ bigarray_start_not_managed arr)

let memcpy_H_to_D_impl ?host_offset ?length ~dst ~src memcpy =
  let full_size = Bigarray.Genarray.size_in_bytes src in
  let elem_bytes = Bigarray.kind_size_in_bytes @@ Bigarray.Genarray.kind src in
  let size_in_bytes =
    match (host_offset, length) with
    | None, None -> full_size
    | Some _, None ->
        invalid_arg "Cudajit.memcpy_H_to_D: providing offset requires providing length"
    | _, Some length -> elem_bytes * length
  in
  let open Ctypes in
  let host =
    match host_offset with
    | None -> get_ptr_not_managed ~reftyp:void src
    | Some offset ->
        let host = get_ptr_not_managed ~reftyp:uint8_t src in
        coerce (ptr uint8_t) (ptr void) @@ (host +@ (offset * elem_bytes))
  in
  memcpy ~dst ~src:host ~size_in_bytes

let memcpy_D_to_H_impl ?host_offset ?length ~dst ~src memcpy =
  let full_size = Bigarray.Genarray.size_in_bytes dst in
  let elem_bytes = Bigarray.kind_size_in_bytes @@ Bigarray.Genarray.kind dst in
  let size_in_bytes =
    match (host_offset, length) with
    | None, None -> full_size
    | Some offset, None -> full_size - (elem_bytes * offset)
    | None, Some length -> elem_bytes * length
    | Some offset, Some length -> elem_bytes * (length - offset)
  in
  let open Ctypes in
  let host =
    match host_offset with
    | None -> get_ptr_not_managed ~reftyp:void dst
    | Some offset ->
        let host = get_ptr_not_managed ~reftyp:uint8_t dst in
        let host = host +@ (offset * elem_bytes) in
        coerce (ptr uint8_t) (ptr void) host
  in
  memcpy ~dst:host ~src ~size_in_bytes

let get_size_in_bytes ?kind ?length ?size_in_bytes provenance =
  match (size_in_bytes, kind, length) with
  | Some size, None, None -> size
  | None, Some kind, Some length ->
      let elem_bytes = Bigarray.kind_size_in_bytes kind in
      elem_bytes * length
  | Some _, Some _, Some _ ->
      invalid_arg @@ provenance
      ^ ": Too many arguments, provide either both [kind] and [length], or just [size_in_bytes]."
  | _ ->
      invalid_arg @@ provenance
      ^ ": Too few arguments, provide either both [kind] and [length], or just [size_in_bytes]."

module Deviceptr = struct
  type t = deviceptr [@@deriving sexp_of]

  let string_of = string_of_deviceptr

  let mem_alloc ~size_in_bytes =
    let open Ctypes in
    let deviceptr = allocate_n cu_deviceptr ~count:1 in
    check "cu_mem_alloc" @@ Cuda.cu_mem_alloc deviceptr @@ Unsigned.Size_t.of_int size_in_bytes;
    Deviceptr !@deviceptr

  let memcpy_H_to_D_unsafe ~dst:(Deviceptr dst) ~(src : unit Ctypes.ptr) ~size_in_bytes =
    check "cu_memcpy_H_to_D" @@ Cuda.cu_memcpy_H_to_D dst src
    @@ Unsigned.Size_t.of_int size_in_bytes

  let memcpy_H_to_D ?host_offset ?length ~dst:(Deviceptr dst) ~src () =
    memcpy_H_to_D_impl ?host_offset ?length ~dst:(Deviceptr dst) ~src memcpy_H_to_D_unsafe

  let alloc_and_memcpy src =
    let size_in_bytes = Bigarray.Genarray.size_in_bytes src in
    let dst = mem_alloc ~size_in_bytes in
    memcpy_H_to_D ~dst ~src ();
    dst

  let memcpy_D_to_H_unsafe ~(dst : unit Ctypes.ptr) ~src:(Deviceptr src) ~size_in_bytes =
    check "cu_memcpy_D_to_H" @@ Cuda.cu_memcpy_D_to_H dst src
    @@ Unsigned.Size_t.of_int size_in_bytes

  let memcpy_D_to_H ?host_offset ?length ~dst ~src () =
    memcpy_D_to_H_impl ?host_offset ?length ~dst ~src memcpy_D_to_H_unsafe

  let memcpy_D_to_D ?kind ?length ?size_in_bytes ~dst:(Deviceptr dst) ~src:(Deviceptr src) () =
    let size_in_bytes = get_size_in_bytes ?kind ?length ?size_in_bytes "memcpy_D_to_D" in
    check "cu_memcpy_D_to_D" @@ Cuda.cu_memcpy_D_to_D dst src
    @@ Unsigned.Size_t.of_int size_in_bytes

  (** Provide either both [kind] and [length], or just [size_in_bytes]. *)
  let memcpy_peer ?kind ?length ?size_in_bytes ~dst:(Deviceptr dst) ~dst_ctx ~src:(Deviceptr src)
      ~src_ctx () =
    let size_in_bytes = get_size_in_bytes ?kind ?length ?size_in_bytes "memcpy_peer" in
    check "cu_memcpy_peer"
    @@ Cuda.cu_memcpy_peer dst dst_ctx src src_ctx
    @@ Unsigned.Size_t.of_int size_in_bytes

  let mem_free (Deviceptr dev) = check "cu_mem_free" @@ Cuda.cu_mem_free dev

  let memset_d8 (Deviceptr dev) v ~length =
    check "cu_memset_d8" @@ Cuda.cu_memset_d8 dev v @@ Unsigned.Size_t.of_int length

  let memset_d16 (Deviceptr dev) v ~length =
    check "cu_memset_d16" @@ Cuda.cu_memset_d16 dev v @@ Unsigned.Size_t.of_int length

  let memset_d32 (Deviceptr dev) v ~length =
    check "cu_memset_d32" @@ Cuda.cu_memset_d32 dev v @@ Unsigned.Size_t.of_int length
end

module Module = struct
  type func = cu_function
  type t = cu_module

  type jit_target =
    | COMPUTE_30
    | COMPUTE_32
    | COMPUTE_35
    | COMPUTE_37
    | COMPUTE_50
    | COMPUTE_52
    | COMPUTE_53
    | COMPUTE_60
    | COMPUTE_61
    | COMPUTE_62
    | COMPUTE_70
    | COMPUTE_72
    | COMPUTE_75
    | COMPUTE_80
    | COMPUTE_86
    | COMPUTE_87
    | COMPUTE_89
    | COMPUTE_90
    | COMPUTE_90A
  [@@deriving sexp]

  type jit_fallback = PREFER_PTX | PREFER_BINARY [@@deriving sexp]
  type jit_cache_mode = NONE | CG | CA [@@deriving sexp]

  (* Note: bool corresponds to C int (0=false). *)
  type jit_option =
    | MAX_REGISTERS of int
    | THREADS_PER_BLOCK of int
    | WALL_TIME of { milliseconds : float }
    | INFO_LOG_BUFFER of (bigstring[@sexp.opaque])
    | ERROR_LOG_BUFFER of (bigstring[@sexp.opaque])
    | OPTIMIZATION_LEVEL of int
    | TARGET_FROM_CUCONTEXT
    | TARGET of jit_target
    | FALLBACK_STRATEGY of jit_fallback
    | GENERATE_DEBUG_INFO of bool
    | LOG_VERBOSE of bool
    | GENERATE_LINE_INFO of bool
    | CACHE_MODE of jit_cache_mode
    | POSITION_INDEPENDENT_CODE of bool
  [@@deriving sexp]

  let cu_jit_target_of = function
    | COMPUTE_30 -> CU_TARGET_COMPUTE_30
    | COMPUTE_32 -> CU_TARGET_COMPUTE_32
    | COMPUTE_35 -> CU_TARGET_COMPUTE_35
    | COMPUTE_37 -> CU_TARGET_COMPUTE_37
    | COMPUTE_50 -> CU_TARGET_COMPUTE_50
    | COMPUTE_52 -> CU_TARGET_COMPUTE_52
    | COMPUTE_53 -> CU_TARGET_COMPUTE_53
    | COMPUTE_60 -> CU_TARGET_COMPUTE_60
    | COMPUTE_61 -> CU_TARGET_COMPUTE_61
    | COMPUTE_62 -> CU_TARGET_COMPUTE_62
    | COMPUTE_70 -> CU_TARGET_COMPUTE_70
    | COMPUTE_72 -> CU_TARGET_COMPUTE_72
    | COMPUTE_75 -> CU_TARGET_COMPUTE_75
    | COMPUTE_80 -> CU_TARGET_COMPUTE_80
    | COMPUTE_86 -> CU_TARGET_COMPUTE_86
    | COMPUTE_87 -> CU_TARGET_COMPUTE_87
    | COMPUTE_89 -> CU_TARGET_COMPUTE_89
    | COMPUTE_90 -> CU_TARGET_COMPUTE_90
    | COMPUTE_90A -> CU_TARGET_COMPUTE_90A

  let cu_jit_fallback_of = function
    | PREFER_PTX -> CU_PREFER_PTX
    | PREFER_BINARY -> CU_PREFER_BINARY

  let cu_jit_cache_mode_of = function
    | NONE -> CU_JIT_CACHE_OPTION_NONE
    | CG -> CU_JIT_CACHE_OPTION_CG
    | CA -> CU_JIT_CACHE_OPTION_CA

  let uint_of_cu_jit_target c =
    let open Cuda_ffi.Types_generated in
    match c with
    | CU_TARGET_COMPUTE_30 -> Unsigned.UInt.of_int64 cu_target_compute_30
    | CU_TARGET_COMPUTE_32 -> Unsigned.UInt.of_int64 cu_target_compute_32
    | CU_TARGET_COMPUTE_35 -> Unsigned.UInt.of_int64 cu_target_compute_35
    | CU_TARGET_COMPUTE_37 -> Unsigned.UInt.of_int64 cu_target_compute_37
    | CU_TARGET_COMPUTE_50 -> Unsigned.UInt.of_int64 cu_target_compute_50
    | CU_TARGET_COMPUTE_52 -> Unsigned.UInt.of_int64 cu_target_compute_52
    | CU_TARGET_COMPUTE_53 -> Unsigned.UInt.of_int64 cu_target_compute_53
    | CU_TARGET_COMPUTE_60 -> Unsigned.UInt.of_int64 cu_target_compute_60
    | CU_TARGET_COMPUTE_61 -> Unsigned.UInt.of_int64 cu_target_compute_61
    | CU_TARGET_COMPUTE_62 -> Unsigned.UInt.of_int64 cu_target_compute_62
    | CU_TARGET_COMPUTE_70 -> Unsigned.UInt.of_int64 cu_target_compute_70
    | CU_TARGET_COMPUTE_72 -> Unsigned.UInt.of_int64 cu_target_compute_72
    | CU_TARGET_COMPUTE_75 -> Unsigned.UInt.of_int64 cu_target_compute_75
    | CU_TARGET_COMPUTE_80 -> Unsigned.UInt.of_int64 cu_target_compute_80
    | CU_TARGET_COMPUTE_86 -> Unsigned.UInt.of_int64 cu_target_compute_86
    | CU_TARGET_COMPUTE_87 -> Unsigned.UInt.of_int64 cu_target_compute_87
    | CU_TARGET_COMPUTE_89 -> Unsigned.UInt.of_int64 cu_target_compute_89
    | CU_TARGET_COMPUTE_90 -> Unsigned.UInt.of_int64 cu_target_compute_90
    | CU_TARGET_COMPUTE_90A -> Unsigned.UInt.of_int64 cu_target_compute_90a
    | CU_TARGET_UNCATEGORIZED c -> Unsigned.UInt.of_int64 c

  let uint_of_cu_jit_fallback c =
    let open Cuda_ffi.Types_generated in
    match c with
    | CU_PREFER_PTX -> Unsigned.UInt.of_int64 cu_prefer_ptx
    | CU_PREFER_BINARY -> Unsigned.UInt.of_int64 cu_prefer_binary
    | CU_PREFER_UNCATEGORIZED c -> Unsigned.UInt.of_int64 c

  let uint_of_cu_jit_cache_mode c =
    let open Cuda_ffi.Types_generated in
    match c with
    | CU_JIT_CACHE_OPTION_NONE -> Unsigned.UInt.of_int64 cu_jit_cache_option_none
    | CU_JIT_CACHE_OPTION_CG -> Unsigned.UInt.of_int64 cu_jit_cache_option_cg
    | CU_JIT_CACHE_OPTION_CA -> Unsigned.UInt.of_int64 cu_jit_cache_option_ca
    | CU_JIT_CACHE_OPTION_UNCATEGORIZED c -> Unsigned.UInt.of_int64 c

  let load_data_ex ptx options =
    let open Ctypes in
    let cu_mod = allocate_n cu_module ~count:1 in
    let n_opts = List.length options in
    let c_options =
      CArray.of_list Cuda_ffi.Types_generated.cu_jit_option
      @@ List.concat_map
           (function
             | MAX_REGISTERS _ -> [ CU_JIT_MAX_REGISTERS ]
             | THREADS_PER_BLOCK _ -> [ CU_JIT_THREADS_PER_BLOCK ]
             | WALL_TIME _ -> [ CU_JIT_WALL_TIME ]
             | INFO_LOG_BUFFER _ -> [ CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES; CU_JIT_INFO_LOG_BUFFER ]
             | ERROR_LOG_BUFFER _ -> [ CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; CU_JIT_ERROR_LOG_BUFFER ]
             | OPTIMIZATION_LEVEL _ -> [ CU_JIT_OPTIMIZATION_LEVEL ]
             | TARGET_FROM_CUCONTEXT -> [ CU_JIT_TARGET_FROM_CUCONTEXT ]
             | TARGET _ -> [ CU_JIT_TARGET ]
             | FALLBACK_STRATEGY _ -> [ CU_JIT_FALLBACK_STRATEGY ]
             | GENERATE_DEBUG_INFO _ -> [ CU_JIT_GENERATE_DEBUG_INFO ]
             | LOG_VERBOSE _ -> [ CU_JIT_LOG_VERBOSE ]
             | GENERATE_LINE_INFO _ -> [ CU_JIT_GENERATE_LINE_INFO ]
             | CACHE_MODE _ -> [ CU_JIT_CACHE_MODE ]
             | POSITION_INDEPENDENT_CODE _ -> [ CU_JIT_POSITION_INDEPENDENT_CODE ])
           options
    in
    let i2u2vp i = coerce (ptr uint) (ptr void) @@ allocate uint @@ Unsigned.UInt.of_int i in
    let u2vp u = coerce (ptr uint) (ptr void) @@ allocate uint u in
    let f2vp f = coerce (ptr float) (ptr void) @@ allocate float f in
    let i2vp i = coerce (ptr int) (ptr void) @@ allocate int i in
    let bi2vp b = coerce (ptr int) (ptr void) @@ allocate int (if b then 1 else 0) in
    let ba2vp b = get_ptr_not_managed ~reftyp:Ctypes_static.void b in
    let c_opts_args =
      CArray.of_list (ptr void)
      @@ List.concat_map
           (function
             | MAX_REGISTERS v -> [ i2u2vp v ]
             | THREADS_PER_BLOCK v -> [ i2u2vp v ]
             | WALL_TIME { milliseconds } -> [ f2vp milliseconds ]
             | INFO_LOG_BUFFER b ->
                 let size = u2vp @@ Unsigned.UInt.of_int @@ Bigarray.Array1.size_in_bytes b in
                 [ size; ba2vp b ]
             | ERROR_LOG_BUFFER b ->
                 let size = u2vp @@ Unsigned.UInt.of_int @@ Bigarray.Array1.size_in_bytes b in
                 [ size; ba2vp b ]
             | OPTIMIZATION_LEVEL i -> [ i2vp i ]
             | TARGET_FROM_CUCONTEXT -> [ null ]
             | TARGET t -> [ u2vp @@ uint_of_cu_jit_target @@ cu_jit_target_of t ]
             | FALLBACK_STRATEGY t -> [ u2vp @@ uint_of_cu_jit_fallback @@ cu_jit_fallback_of t ]
             | GENERATE_DEBUG_INFO c -> [ bi2vp c ]
             | LOG_VERBOSE c -> [ bi2vp c ]
             | GENERATE_LINE_INFO c -> [ bi2vp c ]
             | CACHE_MODE t -> [ u2vp @@ uint_of_cu_jit_cache_mode @@ cu_jit_cache_mode_of t ]
             | POSITION_INDEPENDENT_CODE c -> [ bi2vp c ])
           options
    in
    check "cu_module_load_data_ex"
    @@ Cuda.cu_module_load_data_ex cu_mod
         (coerce (ptr char) (ptr void) ptx.Nvrtc.ptx)
         n_opts (CArray.start c_options)
    @@ CArray.start c_opts_args;
    !@cu_mod

  let get_function module_ ~name =
    let open Ctypes in
    let func = allocate_n cu_function ~count:1 in
    check "cu_module_get_function" @@ Cuda.cu_module_get_function func module_ name;
    !@func

  let get_global module_ ~name =
    let open Ctypes in
    let device = allocate_n cu_deviceptr ~count:1 in
    let size_in_bytes = allocate size_t Unsigned.Size_t.zero in
    check "cu_module_get_global" @@ Cuda.cu_module_get_global device size_in_bytes module_ name;
    (Deviceptr !@device, !@size_in_bytes)

  let unload cu_mod = check "cu_module_unload" @@ Cuda.cu_module_unload cu_mod
end

module Stream = struct
  type t = stream

  let memcpy_H_to_D_unsafe ~dst:(Deviceptr dst) ~(src : unit Ctypes.ptr) ~size_in_bytes stream =
    check "cu_memcpy_H_to_D_async"
    @@ Cuda.cu_memcpy_H_to_D_async dst src (Unsigned.Size_t.of_int size_in_bytes) stream.stream

  let memcpy_H_to_D ?host_offset ?length ~dst:(Deviceptr dst) ~src =
    memcpy_H_to_D_impl ?host_offset ?length ~dst:(Deviceptr dst) ~src memcpy_H_to_D_unsafe

  type size_t = Unsigned.size_t

  let sexp_of_size_t i = Sexplib0.Sexp.Atom (Unsigned.Size_t.to_string i)

  type kernel_param =
    | Tensor of Deviceptr.t
    | Int of int
    | Size_t of size_t
    | Single of float
    | Double of float
  [@@deriving sexp_of]

  let no_stream =
    { args_lifetimes = []; owned_events = []; stream = Ctypes.(coerce (ptr void) cu_stream null) }

  let launch_kernel func ~grid_dim_x ?(grid_dim_y = 1) ?(grid_dim_z = 1) ~block_dim_x
      ?(block_dim_y = 1) ?(block_dim_z = 1) ~shared_mem_bytes stream kernel_params =
    let i2u = Unsigned.UInt.of_int in
    let open Ctypes in
    let kernel_params =
      List.map
        (function
          | Tensor (Deviceptr dev) -> coerce (ptr uint64_t) (ptr void) @@ allocate uint64_t dev
          | Int i -> coerce (ptr int) (ptr void) @@ allocate int i
          | Size_t u -> coerce (ptr size_t) (ptr void) @@ allocate size_t u
          | Single u -> coerce (ptr float) (ptr void) @@ allocate float u
          | Double u -> coerce (ptr double) (ptr void) @@ allocate double u)
        kernel_params
    in
    let c_kernel_params = kernel_params |> CArray.of_list (ptr void) in
    check "cu_launch_kernel"
    @@ Cuda.cu_launch_kernel func (i2u grid_dim_x) (i2u grid_dim_y) (i2u grid_dim_z)
         (i2u block_dim_x) (i2u block_dim_y) (i2u block_dim_z) (i2u shared_mem_bytes) stream.stream
         (CArray.start c_kernel_params)
    @@ coerce (ptr void) (ptr @@ ptr void) null;
    stream.args_lifetimes <- Remember (kernel_params, c_kernel_params) :: stream.args_lifetimes

  type attach_mem = GLOBAL | HOST | SINGLE_stream [@@deriving sexp]

  let uint_of_attach_mem f =
    let open Cuda_ffi.Types_generated in
    match f with
    | GLOBAL -> Unsigned.UInt.of_int64 cu_mem_attach_global
    | HOST -> Unsigned.UInt.of_int64 cu_mem_attach_host
    | SINGLE_stream -> Unsigned.UInt.of_int64 cu_mem_attach_single

  let attach_mem stream (Deviceptr device) length flag =
    check "cu_stream_attach_mem_async"
    @@ Cuda.cu_stream_attach_mem_async stream.stream device (Unsigned.Size_t.of_int length)
    @@ uint_of_attach_mem flag

  let uint_of_cu_stream_flags ~non_blocking =
    let open Cuda_ffi.Types_generated in
    match non_blocking with
    | false -> Unsigned.UInt.of_int64 cu_stream_default
    | true -> Unsigned.UInt.of_int64 cu_stream_non_blocking

  let destroy stream =
    release_stream stream;
    check "cu_stream_destroy" @@ Cuda.cu_stream_destroy stream.stream

  let create ?(non_blocking = false) ?(lower_priority = 0) () =
    let open Ctypes in
    let stream = allocate_n cu_stream ~count:1 in
    check "cu_stream_create_with_priority"
    @@ Cuda.cu_stream_create_with_priority stream
         (uint_of_cu_stream_flags ~non_blocking)
         lower_priority;
    let stream = { args_lifetimes = []; owned_events = []; stream = !@stream } in
    Stdlib.Gc.finalise destroy stream;
    stream

  let get_context stream =
    let open Ctypes in
    let ctx = allocate_n cu_context ~count:1 in
    check "cu_stream_get_ctx" @@ Cuda.cu_stream_get_ctx stream.stream ctx;
    !@ctx

  let get_id stream =
    let open Ctypes in
    let id = allocate uint64_t Unsigned.UInt64.zero in
    check "cu_stream_get_id" @@ Cuda.cu_stream_get_id stream.stream id;
    !@id

  let is_ready stream =
    match Cuda.cu_stream_query stream.stream with
    | CUDA_ERROR_NOT_READY -> false
    | e ->
        check "cu_stream_query" e;
        (* We do not destroy delimited events, but any kernel arguments no longer needed. *)
        stream.args_lifetimes <- [];
        true

  let synchronize stream =
    check "cu_stream_synchronize" @@ Cuda.cu_stream_synchronize stream.stream;
    release_stream stream

  let memcpy_D_to_H_unsafe ~(dst : unit Ctypes.ptr) ~src:(Deviceptr src) ~size_in_bytes stream =
    check "cu_memcpy_D_to_H_async"
    @@ Cuda.cu_memcpy_D_to_H_async dst src (Unsigned.Size_t.of_int size_in_bytes) stream.stream

  let memcpy_D_to_H ?host_offset ?length ~dst ~src =
    memcpy_D_to_H_impl ?host_offset ?length ~dst ~src memcpy_D_to_H_unsafe

  let memcpy_D_to_D ?kind ?length ?size_in_bytes ~dst:(Deviceptr dst) ~src:(Deviceptr src) stream =
    let size_in_bytes = get_size_in_bytes ?kind ?length ?size_in_bytes "memcpy_D_to_D_async" in
    check "cu_memcpy_D_to_D_async"
    @@ Cuda.cu_memcpy_D_to_D_async dst src (Unsigned.Size_t.of_int size_in_bytes) stream.stream

  (** Provide either both [kind] and [length], or just [size_in_bytes]. *)
  let memcpy_peer ?kind ?length ?size_in_bytes ~dst:(Deviceptr dst) ~dst_ctx ~src:(Deviceptr src)
      ~src_ctx stream =
    let size_in_bytes = get_size_in_bytes ?kind ?length ?size_in_bytes "memcpy_peer_async" in
    check "cu_memcpy_peer_async"
    @@ Cuda.cu_memcpy_peer_async dst dst_ctx src src_ctx
         (Unsigned.Size_t.of_int size_in_bytes)
         stream.stream

  let memset_d8 (Deviceptr dev) v ~length stream =
    check "cu_memset_d8_async"
    @@ Cuda.cu_memset_d8_async dev v (Unsigned.Size_t.of_int length) stream.stream

  let memset_d16 (Deviceptr dev) v ~length stream =
    check "cu_memset_d16_async"
    @@ Cuda.cu_memset_d16_async dev v (Unsigned.Size_t.of_int length) stream.stream

  let memset_d32 (Deviceptr dev) v ~length stream =
    check "cu_memset_d32_async"
    @@ Cuda.cu_memset_d32_async dev v (Unsigned.Size_t.of_int length) stream.stream
end

module Event = struct
  type t = cu_event

  let sexp_of_t (event : t) = sexp_of_voidp @@ Ctypes.to_voidp event

  let uint_of_cu_event_flags ~blocking_sync ~enable_timing ~interprocess =
    let open Cuda_ffi.Types_generated in
    let open Unsigned.UInt in
    let default = Unsigned.UInt.of_int64 cu_event_default in
    let blocking_sync = if blocking_sync then of_int64 cu_event_blocking_sync else zero in
    let disable_timing = if enable_timing then zero else of_int64 cu_event_disable_timing in
    let interprocess = if interprocess then of_int64 cu_event_interprocess else zero in
    List.fold_left
      (fun flags flag -> Infix.(flags lor flag))
      default
      [ blocking_sync; disable_timing; interprocess ]

  let destroy event = destroy_event event

  let create ?(blocking_sync = false) ?(enable_timing = false) ?(interprocess = false) () =
    let open Ctypes in
    let event = allocate_n cu_event ~count:1 in
    check "cu_event_create"
    @@ Cuda.cu_event_create event
         (uint_of_cu_event_flags ~blocking_sync ~enable_timing ~interprocess);
    let event = !@event in
    Gc.finalise destroy event;
    event

  let elapsed_time ~start ~end_ =
    let open Ctypes in
    let result = allocate float 0.0 in
    check "cu_event_elapsed_time" @@ Cuda.cu_event_elapsed_time result start end_;
    !@result

  let query event =
    match Cuda.cu_event_query event with
    | CUDA_SUCCESS -> true
    | CUDA_ERROR_NOT_READY -> false
    | error ->
        check "cu_event_query" error;
        false

  let record ?(external_ = false) event stream =
    let open Cuda_ffi.Types_generated in
    let flags =
      Unsigned.UInt.of_int64
      @@ if external_ then cu_event_record_default else cu_event_record_external
    in
    check "cu_event_record_with_flags" @@ Cuda.cu_event_record_with_flags event stream.stream flags

  let synchronize event = check "cu_event_synchronize" @@ Cuda.cu_event_synchronize event

  let wait ?(external_ = false) stream event =
    let open Cuda_ffi.Types_generated in
    let flags =
      Unsigned.UInt.of_int64 @@ if external_ then cu_event_wait_default else cu_event_wait_external
    in
    check "cu_stream_wait_event" @@ Cuda.cu_stream_wait_event stream.stream event flags
end

module Delimited_event = struct
  type t = delimited_event

  let query event = if event.is_released then true else Event.query event.event

  let record ?blocking_sync ?interprocess ?external_ stream =
    let event = Event.create ?blocking_sync ~enable_timing:false ?interprocess () in
    Event.record ?external_ event stream;
    let result = { event; is_released = false } in
    stream.owned_events <- result :: stream.owned_events;
    result

  let synchronize event =
    if not event.is_released then (
      Event.synchronize event.event;
      destroy_event event.event;
      event.is_released <- true)

  let wait ?external_ stream event =
    if not event.is_released then Event.wait ?external_ stream event.event

  let is_released event = event.is_released
end
