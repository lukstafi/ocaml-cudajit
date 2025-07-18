# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is ocaml-cudajit, an OCaml binding library for NVIDIA CUDA runtime compilation and kernel execution. The project provides a unified interface to both the CUDA driver API and NVRTC (NVIDIA Runtime Compilation) library, enabling compilation and execution of CUDA kernels from OCaml.

## Core Architecture

The library is structured in three main layers:

1. **FFI Bindings Layer** (`cuda_ffi/`, `nvrtc_ffi/`): Low-level C bindings using ocaml-ctypes
   - `cuda_ffi/` - Direct bindings to CUDA driver API
   - `nvrtc_ffi/` - Direct bindings to NVRTC library
   - These generate type-safe OCaml bindings from C headers

2. **High-Level API Layer** (`src/`): User-friendly OCaml modules
   - `cuda.ml/mli` - Main CUDA interface with modules for Device, Context, Module, Stream, Event, etc.
   - `nvrtc.ml/mli` - NVRTC interface for runtime compilation of CUDA kernels

3. **Application Layer** (`bin/`, `test/`, `test_no_device/`): Examples and tests
   - `test/saxpy.ml` - Full CUDA workflow test requiring GPU
   - `test_no_device/saxpy_ptx.ml` - NVRTC compilation test (no GPU required)

## Build System

The project uses Dune with specific considerations for CUDA:

- **CUDA Path Detection**: Automatically detects CUDA installation path via `CUDA_PATH` environment variable
- **Platform-Specific Linking**: Different library linking for Windows vs Unix systems
- **Cross-Platform Support**: Handles Windows-specific symlink creation for CUDA paths

## Common Development Commands

- `dune build` - Build the entire project
- `dune exec test/saxpy.exe` - Run full CUDA test (requires GPU)
- `dune exec test_no_device/saxpy_ptx.exe` - Run NVRTC compilation test (no GPU)
- `dune runtest` - Run all tests
- `dune clean` - Clean build artifacts

## Key Design Patterns

- **Resource Management**: Automatic cleanup of CUDA resources using OCaml finalizers
- **Type Safety**: Extensive use of phantom types and modules to prevent misuse
- **Error Handling**: CUDA error codes converted to OCaml exceptions with descriptive messages
- **Memory Management**: Safe bigarray integration for host-device memory transfers

## Testing Strategy

Two test categories:
1. **GPU Tests** (`test/`): Full CUDA workflow tests requiring actual GPU hardware
2. **No-GPU Tests** (`test_no_device/`): NVRTC compilation tests that can run without GPU

## Dependencies

- `ctypes` and `ctypes-foreign` for C bindings
- `sexplib0` for serialization
- `conf-cuda` for CUDA system detection
- Requires CUDA toolkit installation on system

## Module Hierarchy

- `Cuda` module provides: `Device`, `Context`, `Module`, `Stream`, `Event`, `Delimited_event`, `Deviceptr`
- `Nvrtc` module provides: runtime compilation functions for CUDA C++ to PTX
- FFI modules auto-generated from C headers using ctypes

## Working with CUDA Code

The typical workflow involves:

1. Write CUDA C++ kernel as OCaml string
2. Use `Nvrtc.compile_to_ptx` to compile to PTX assembly
3. Load PTX using `Cuda.Module.load_data_ex`
4. Execute kernels using `Cuda.Stream.launch_kernel`

## Important Notes

- CUDA context management is handled automatically through OCaml's resource management
- The library supports both synchronous and asynchronous CUDA operations
- Memory transfers between host and device use OCaml bigarrays for type safety
- Error handling converts CUDA status codes to meaningful OCaml exceptions
