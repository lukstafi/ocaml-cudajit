open Ctypes
open Bindings_types

module Functions (F : Ctypes.FOREIGN) = struct
  module E = Types_generated

  let cu_init = F.foreign "cuInit" F.(int @-> returning E.cu_result)
  let cu_device_get_count = F.foreign "cuDeviceGetCount" F.(ptr int @-> returning E.cu_result)
  let cu_device_get = F.foreign "cuDeviceGet" F.(ptr E.cu_device @-> int @-> returning E.cu_result)

  let cu_ctx_create =
    F.foreign "cuCtxCreate" F.(ptr cu_context @-> uint @-> E.cu_device @-> returning E.cu_result)

  let cu_ctx_get_flags = F.foreign "cuCtxGetFlags" F.(ptr uint @-> returning E.cu_result)

  let cu_device_primary_ctx_retain =
    F.foreign "cuDevicePrimaryCtxRetain"
      F.(ptr cu_context @-> E.cu_device @-> returning E.cu_result)

  let cu_device_primary_ctx_release =
    F.foreign "cuDevicePrimaryCtxRelease" F.(E.cu_device @-> returning E.cu_result)

  let cu_device_primary_ctx_reset =
    F.foreign "cuDevicePrimaryCtxReset" F.(E.cu_device @-> returning E.cu_result)

  let cu_ctx_get_device = F.foreign "cuCtxGetDevice" F.(ptr E.cu_device @-> returning E.cu_result)
  let cu_ctx_get_current = F.foreign "cuCtxGetCurrent" F.(ptr cu_context @-> returning E.cu_result)
  let cu_ctx_pop_current = F.foreign "cuCtxPopCurrent" F.(ptr cu_context @-> returning E.cu_result)
  let cu_ctx_set_current = F.foreign "cuCtxSetCurrent" F.(cu_context @-> returning E.cu_result)
  let cu_ctx_push_current = F.foreign "cuCtxPushCurrent" F.(cu_context @-> returning E.cu_result)

  let cu_module_load_data_ex =
    F.foreign "cuModuleLoadDataEx"
      F.(
        ptr cu_module @-> ptr void @-> int @-> ptr E.cu_jit_option
        @-> ptr (ptr void)
        @-> returning E.cu_result)

  let cu_module_get_function =
    F.foreign "cuModuleGetFunction"
      F.(ptr cu_function @-> cu_module @-> string @-> returning E.cu_result)

  let cu_mem_alloc =
    F.foreign "cuMemAlloc" F.(ptr cu_deviceptr @-> size_t @-> returning E.cu_result)

  let cu_mem_alloc_async =
    F.foreign "cuMemAllocAsync"
      F.(ptr cu_deviceptr @-> size_t @-> cu_stream @-> returning E.cu_result)

  let cu_memcpy_H_to_D =
    F.foreign "cuMemcpyHtoD" F.(cu_deviceptr @-> ptr void @-> size_t @-> returning E.cu_result)

  let cu_memcpy_H_to_D_async =
    F.foreign "cuMemcpyHtoDAsync"
      F.(cu_deviceptr @-> ptr void @-> size_t @-> cu_stream @-> returning E.cu_result)

  let cu_launch_kernel =
    F.foreign "cuLaunchKernel"
      F.(
        cu_function @-> uint @-> uint @-> uint @-> uint @-> uint @-> uint @-> uint @-> cu_stream
        @-> ptr (ptr void)
        @-> ptr (ptr void)
        @-> returning E.cu_result)

  let cu_ctx_synchronize = F.foreign "cuCtxSynchronize" F.(void @-> returning E.cu_result)

  let cu_memcpy_D_to_H =
    F.foreign "cuMemcpyDtoH" F.(ptr void @-> cu_deviceptr @-> size_t @-> returning E.cu_result)

  let cu_memcpy_D_to_H_async =
    F.foreign "cuMemcpyDtoHAsync"
      F.(ptr void @-> cu_deviceptr @-> size_t @-> cu_stream @-> returning E.cu_result)

  let cu_memcpy_D_to_D =
    F.foreign "cuMemcpyDtoD" F.(cu_deviceptr @-> cu_deviceptr @-> size_t @-> returning E.cu_result)

  let cu_memcpy_D_to_D_async =
    F.foreign "cuMemcpyDtoDAsync"
      F.(cu_deviceptr @-> cu_deviceptr @-> size_t @-> cu_stream @-> returning E.cu_result)

  let cu_memcpy_peer =
    F.foreign "cuMemcpyPeer"
      F.(
        cu_deviceptr @-> cu_context @-> cu_deviceptr @-> cu_context @-> size_t
        @-> returning E.cu_result)

  let cu_memcpy_peer_async =
    F.foreign "cuMemcpyPeerAsync"
      F.(
        cu_deviceptr @-> cu_context @-> cu_deviceptr @-> cu_context @-> size_t @-> cu_stream
        @-> returning E.cu_result)

  let cu_ctx_disable_peer_access =
    F.foreign "cuCtxDisablePeerAccess" F.(cu_context @-> returning E.cu_result)

  let cu_ctx_enable_peer_access =
    F.foreign "cuCtxEnablePeerAccess" F.(cu_context @-> uint @-> returning E.cu_result)

  let cu_device_can_access_peer =
    F.foreign "cuDeviceCanAccessPeer"
      F.(ptr int @-> E.cu_device @-> E.cu_device @-> returning E.cu_result)

  let cu_device_get_p2p_attribute =
    F.foreign "cuDeviceGetP2PAttribute"
      F.(
        ptr int @-> E.cu_device_p2p_attribute @-> E.cu_device @-> E.cu_device
        @-> returning E.cu_result)

  let cu_mem_free = F.foreign "cuMemFree" F.(cu_deviceptr @-> returning E.cu_result)

  let cu_mem_free_async =
    F.foreign "cuMemFreeAsync" F.(cu_deviceptr @-> cu_stream @-> returning E.cu_result)

  let cu_module_unload = F.foreign "cuModuleUnload" F.(cu_module @-> returning E.cu_result)
  let cu_ctx_destroy = F.foreign "cuCtxDestroy" F.(cu_context @-> returning E.cu_result)

  let cu_memset_d8 =
    F.foreign "cuMemsetD8" F.(cu_deviceptr @-> uchar @-> size_t @-> returning E.cu_result)

  let cu_memset_d16 =
    F.foreign "cuMemsetD16" F.(cu_deviceptr @-> ushort @-> size_t @-> returning E.cu_result)

  let cu_memset_d32 =
    F.foreign "cuMemsetD32" F.(cu_deviceptr @-> uint32_t @-> size_t @-> returning E.cu_result)

  let cu_memset_d8_async =
    F.foreign "cuMemsetD8Async"
      F.(cu_deviceptr @-> uchar @-> size_t @-> cu_stream @-> returning E.cu_result)

  let cu_memset_d16_async =
    F.foreign "cuMemsetD16Async"
      F.(cu_deviceptr @-> ushort @-> size_t @-> cu_stream @-> returning E.cu_result)

  let cu_memset_d32_async =
    F.foreign "cuMemsetD32Async"
      F.(cu_deviceptr @-> uint32_t @-> size_t @-> cu_stream @-> returning E.cu_result)

  let cu_mem_get_info =
    F.foreign "cuMemGetInfo" F.(ptr size_t @-> ptr size_t @-> returning E.cu_result)

  let cu_module_get_global =
    F.foreign "cuModuleGetGlobal_v2"
      F.(ptr cu_deviceptr @-> ptr size_t @-> cu_module @-> string @-> returning E.cu_result)

  let cu_device_get_name =
    F.foreign "cuDeviceGetName" F.(ptr char @-> int @-> E.cu_device @-> returning E.cu_result)

  let cu_device_get_attribute =
    F.foreign "cuDeviceGetAttribute"
      F.(ptr int @-> E.cu_device_attribute @-> E.cu_device @-> returning E.cu_result)

  let cu_computemode_of_int mode =
    if mode = Int64.to_int E.cu_computemode_default then CU_COMPUTEMODE_DEFAULT
    else if mode = Int64.to_int E.cu_computemode_exclusive_process then
      CU_COMPUTEMODE_EXCLUSIVE_PROCESS
    else if mode = Int64.to_int E.cu_computemode_prohibited then CU_COMPUTEMODE_PROHIBITED
    else CU_COMPUTEMODE_UNCATEGORIZED (Int64.of_int mode)

  let cu_ctx_set_limit =
    F.foreign "cuCtxSetLimit" F.(E.cu_limit @-> size_t @-> returning E.cu_result)

  let cu_ctx_get_limit =
    F.foreign "cuCtxGetLimit" F.(ptr size_t @-> E.cu_limit @-> returning E.cu_result)

  let cu_stream_attach_mem_async =
    F.foreign "cuStreamAttachMemAsync"
      F.(cu_stream @-> cu_deviceptr @-> size_t @-> uint @-> returning E.cu_result)

  let cu_stream_create_with_priority =
    F.foreign "cuStreamCreateWithPriority"
      F.(ptr cu_stream @-> uint @-> int @-> returning E.cu_result)

  let cu_stream_destroy = F.foreign "cuStreamDestroy" F.(cu_stream @-> returning E.cu_result)

  let cu_stream_get_ctx =
    F.foreign "cuStreamGetCtx" F.(cu_stream @-> ptr cu_context @-> returning E.cu_result)

  let cu_stream_get_id =
    F.foreign "cuStreamGetId" F.(cu_stream @-> ptr uint64_t @-> returning E.cu_result)

  let cu_stream_query = F.foreign "cuStreamQuery" F.(cu_stream @-> returning E.cu_result)

  let cu_stream_synchronize =
    F.foreign "cuStreamSynchronize" F.(cu_stream @-> returning E.cu_result)

  let cu_event_create =
    F.foreign "cuEventCreate" F.(ptr cu_event @-> uint @-> returning E.cu_result)

  let cu_event_destroy = F.foreign "cuEventDestroy" F.(cu_event @-> returning E.cu_result)

  let cu_event_elapsed_time =
    F.foreign "cuEventElapsedTime" F.(ptr float @-> cu_event @-> cu_event @-> returning E.cu_result)

  let cu_event_record_with_flags =
    F.foreign "cuEventRecordWithFlags" F.(cu_event @-> cu_stream @-> uint @-> returning E.cu_result)

  let cu_event_query = F.foreign "cuEventQuery" F.(cu_event @-> returning E.cu_result)
  let cu_event_synchronize = F.foreign "cuEventSynchronize" F.(cu_event @-> returning E.cu_result)

  let cu_stream_wait_event =
    F.foreign "cuStreamWaitEvent" F.(cu_stream @-> cu_event @-> uint @-> returning E.cu_result)
end
