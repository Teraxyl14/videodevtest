import os
import torch
import torch.multiprocessing as mp

def initialize_persistent_hub_buffer(buffer_size_mb: int = 100) -> tuple:
    """
    Allocates a persistent, contiguous block of VRAM in the Hub process.
    Generates the 64-byte IPC handle required for zero-copy Spoke access.
    """
    # Allocate uninitialized VRAM directly on the GPU
    buffer_bytes = buffer_size_mb * 1024 * 1024
    shared_cuda_tensor = torch.empty(buffer_bytes, dtype=torch.uint8, device='cuda')
    
    # Extract the CUDA IPC handle via PyTorch reductions (returns a rebuild function and arguments)
    ipc_handle = torch.multiprocessing.reductions.reduce_tensor(shared_cuda_tensor)
    
    return shared_cuda_tensor, ipc_handle

def ephemeral_spoke_ingestion_protocol(ipc_handle: tuple) -> torch.Tensor:
    """
    Executed strictly within the isolated Spoke process.
    Reconstructs the PyTorch tensor mapping to the exact physical VRAM
    without initiating a host-to-device memory transfer.
    """
    # Rebuild the tensor utilizing cudaIpcOpenMemHandle under the hood
    rebuild_function, rebuild_args = ipc_handle[0], ipc_handle[1]
    spoke_tensor = rebuild_function(*rebuild_args)
    
    return spoke_tensor
