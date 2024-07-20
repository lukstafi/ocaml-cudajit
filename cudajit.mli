type context
type func
type module_
type limit
type device
type nvrtc_result [@@deriving sexp]
type cuda_result [@@deriving sexp]
type error_code = Nvrtc_error of nvrtc_result | Cuda_error of cuda_result [@@deriving sexp]

exception Error of { status : error_code; message : string }

type compile_to_ptx_result

val compile_to_ptx :
  cu_src:string -> name:string -> options:string list -> with_debug:bool -> compile_to_ptx_result

val string_from_ptx : compile_to_ptx_result -> string
val compilation_log : compile_to_ptx_result -> string option
val init : ?flags:int -> unit -> unit
val device_get_count : unit -> int
val device_get : ordinal:int -> device

type ctx_flag =
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

type ctx_flags = ctx_flag list [@@deriving sexp]

val ctx_create : ctx_flags -> device -> context
val ctx_get_flags : unit -> ctx_flags
val device_primary_ctx_release : device -> unit
val device_primary_ctx_reset : device -> unit
val device_primary_ctx_retain : device -> context
val ctx_get_device : unit -> device
val ctx_pop_current : unit -> context
val ctx_get_current : unit -> context
val ctx_push_current : context -> unit
val ctx_set_current : context -> unit

type bigstring = (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

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

type jit_option =
  | JIT_MAX_REGISTERS of int
  | JIT_THREADS_PER_BLOCK of int
  | JIT_WALL_TIME of { milliseconds : float }
  | JIT_INFO_LOG_BUFFER of bigstring
  | JIT_ERROR_LOG_BUFFER of bigstring
  | JIT_OPTIMIZATION_LEVEL of int
  | JIT_TARGET_FROM_CUCONTEXT
  | JIT_TARGET of jit_target
  | JIT_FALLBACK_STRATEGY of jit_fallback
  | JIT_GENERATE_DEBUG_INFO of bool
  | JIT_LOG_VERBOSE of bool
  | JIT_GENERATE_LINE_INFO of bool
  | JIT_CACHE_MODE of jit_cache_mode
  | JIT_POSITION_INDEPENDENT_CODE of bool
[@@deriving sexp]

val module_load_data_ex : compile_to_ptx_result -> jit_option list -> module_
val module_get_function : module_ -> name:string -> func

type deviceptr [@@deriving sexp_of]

val string_of_deviceptr : deviceptr -> string
val mem_alloc : size_in_bytes:int -> deviceptr
val memcpy_H_to_D_unsafe : dst:deviceptr -> src:unit Ctypes.ptr -> size_in_bytes:int -> unit

val memcpy_H_to_D :
  ?host_offset:int ->
  ?length:int ->
  dst:deviceptr ->
  src:('a, 'b, 'c) Bigarray.Genarray.t ->
  unit ->
  unit

val alloc_and_memcpy : ('a, 'b, 'c) Bigarray.Genarray.t -> deviceptr

type stream

val memcpy_H_to_D_async_unsafe :
  dst:deviceptr -> src:unit Ctypes.ptr -> size_in_bytes:int -> stream -> unit

val memcpy_H_to_D_async :
  ?host_offset:int ->
  ?length:int ->
  dst:deviceptr ->
  src:('a, 'b, 'c) Bigarray.Genarray.t ->
  stream ->
  unit

type kernel_param =
  | Tensor of deviceptr
  | Int of int
  | Size_t of Unsigned.size_t
  | Single of float
  | Double of float

val no_stream : stream

val launch_kernel :
  func ->
  grid_dim_x:int ->
  ?grid_dim_y:int ->
  ?grid_dim_z:int ->
  block_dim_x:int ->
  ?block_dim_y:int ->
  ?block_dim_z:int ->
  shared_mem_bytes:int ->
  stream ->
  kernel_param list ->
  unit

val ctx_synchronize : unit -> unit
val memcpy_D_to_H_unsafe : dst:unit Ctypes.ptr -> src:deviceptr -> size_in_bytes:int -> unit

val memcpy_D_to_H :
  ?host_offset:int ->
  ?length:int ->
  dst:('a, 'b, 'c) Bigarray.Genarray.t ->
  src:deviceptr ->
  unit ->
  unit

val memcpy_D_to_H_async_unsafe :
  dst:unit Ctypes.ptr -> src:deviceptr -> size_in_bytes:int -> stream -> unit

val memcpy_D_to_H_async :
  ?host_offset:int ->
  ?length:int ->
  dst:('a, 'b, 'c) Bigarray.Genarray.t ->
  src:deviceptr ->
  stream ->
  unit

val get_size_in_bytes :
  ?kind:('a, 'b) Bigarray.kind -> ?length:int -> ?size_in_bytes:int -> string -> int

val memcpy_D_to_D :
  ?kind:('a, 'b) Bigarray.kind ->
  ?length:int ->
  ?size_in_bytes:int ->
  dst:deviceptr ->
  src:deviceptr ->
  unit ->
  unit

val memcpy_D_to_D_async :
  ?kind:('a, 'b) Bigarray.kind ->
  ?length:int ->
  ?size_in_bytes:int ->
  dst:deviceptr ->
  src:deviceptr ->
  stream ->
  unit

val memcpy_peer :
  ?kind:('a, 'b) Bigarray.kind ->
  ?length:int ->
  ?size_in_bytes:int ->
  dst:deviceptr ->
  dst_ctx:context ->
  src:deviceptr ->
  src_ctx:context ->
  unit ->
  unit

val memcpy_peer_async :
  ?kind:('a, 'b) Bigarray.kind ->
  ?length:int ->
  ?size_in_bytes:int ->
  dst:deviceptr ->
  dst_ctx:context ->
  src:deviceptr ->
  src_ctx:context ->
  stream ->
  unit

val ctx_disable_peer_access : context -> unit
val ctx_enable_peer_access : ?flags:Unsigned.uint -> context -> unit
val device_can_access_peer : dst:deviceptr -> src:deviceptr -> bool

type p2p_attribute =
  | PERFORMANCE_RANK of int
  | ACCESS_SUPPORTED of bool
  | NATIVE_ATOMIC_SUPPORTED of bool
  | CUDA_ARRAY_ACCESS_SUPPORTED of bool

val device_get_p2p_attributes : dst:deviceptr -> src:deviceptr -> p2p_attribute list
val mem_free : deviceptr -> unit
val module_unload : module_ -> unit
val ctx_destroy : context -> unit
val memset_d8 : deviceptr -> Unsigned.uchar -> length:int -> unit
val memset_d16 : deviceptr -> Unsigned.ushort -> length:int -> unit
val memset_d32 : deviceptr -> Unsigned.uint32 -> length:int -> unit
val memset_d8_async : deviceptr -> Unsigned.uchar -> length:int -> stream -> unit
val memset_d16_async : deviceptr -> Unsigned.ushort -> length:int -> stream -> unit
val memset_d32_async : deviceptr -> Unsigned.uint32 -> length:int -> stream -> unit
val module_get_global : module_ -> name:string -> deviceptr * Unsigned.size_t

type computemode = DEFAULT | PROHIBITED | EXCLUSIVE_PROCESS [@@deriving sexp]
type flush_GPU_direct_RDMA_writes_options = HOST | MEMOPS [@@deriving sexp]

type device_attributes = {
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
  mempool_supported_handle_types : bool;
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

val device_get_attributes : device -> device_attributes
val ctx_set_limit : limit -> int -> unit
val ctx_get_limit : limit -> Unsigned.size_t

type attach_mem = Mem_global | Mem_host | Mem_single_stream [@@deriving sexp]

val stream_attach_mem_async : stream -> deviceptr -> int -> attach_mem -> unit
val stream_create : ?non_blocking:bool -> ?lower_priority:int -> unit -> stream
val stream_destroy : stream -> unit
val stream_get_context : stream -> context
val stream_get_id : stream -> Unsigned.uint64
val stream_is_ready : stream -> bool
val stream_synchronize : stream -> unit
