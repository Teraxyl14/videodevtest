import torch
import pynvml
import sys

def check_memory():
    print("=== VRAM DIAGNOSTIC ===")
    
    # Check NVML (System Level)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"NVML Total: {info.total / 1024**3:.2f} GB")
        print(f"NVML Used:  {info.used / 1024**3:.2f} GB")
        print(f"NVML Free:  {info.free / 1024**3:.2f} GB")
    except Exception as e:
        print(f"NVML Error: {e}")

    # Check PyTorch
    if torch.cuda.is_available():
        print(f"\nPyTorch Device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"PyTorch Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # Try emptying cache
        print("\nEmptying Cache...")
        torch.cuda.empty_cache()
        print(f"Post-Empty Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Post-Empty Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        print("PyTorch CUDA not available")

if __name__ == "__main__":
    check_memory()
