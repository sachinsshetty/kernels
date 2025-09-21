import torch

def verify_gpu_specs():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure an NVIDIA GPU is installed and PyTorch is configured with CUDA support.")
        return

    # Get the current CUDA device
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"GPU: {props.name}")
    print("\nVerifying RTX 4050 Laptop GPU Specifications:\n")

    # Provided specifications
    expected_specs = {
        "Shading Units": 2560,
        "TMUs": 80,
        "ROPs": 32,
        "SM Count": 18,
        "Tensor Cores": 120,
        "RT Cores": 18,
        "L1 Cache per SM": "128 KB",
        "L2 Cache": "32 MB"
    }

    # 1. SM Count
    sm_count = props.multi_processor_count
    print(f"SM Count: {sm_count} (Expected: {expected_specs['SM Count']})")
    print(f"SM Count Match: {sm_count == expected_specs['SM Count']}")

    # 2. Shading Units (CUDA Cores)
    # Ada Lovelace architecture: 128 CUDA cores per SM
    cuda_cores_per_sm = 128  # Based on NVIDIA Ada Lovelace architecture
    shading_units = sm_count * cuda_cores_per_sm
    print(f"Shading Units (CUDA Cores): {shading_units} (Expected: {expected_specs['Shading Units']})")
    print(f"Shading Units Match: {shading_units == expected_specs['Shading Units']}")

    # 3. TMUs
    # TMUs are not directly exposed by CUDA APIs. Use expected value.
    print(f"TMUs: Cannot query directly via PyTorch/CUDA. Expected: {expected_specs['TMUs']}")
    print("Note: TMUs require NVIDIA documentation or tools like NVIDIA Nsight for verification.")

    # 4. ROPs
    # ROPs are not directly exposed by CUDA APIs. Use expected value.
    print(f"ROPs: Cannot query directly via PyTorch/CUDA. Expected: {expected_specs['ROPs']}")
    print("Note: ROPs require NVIDIA documentation or tools like NVIDIA Nsight for verification.")

    # 5. Tensor Cores
    # Ada Lovelace: 4 Tensor Cores per SM
    tensor_cores_per_sm = 4  # Based on Ada Lovelace architecture
    tensor_cores = sm_count * tensor_cores_per_sm
    print(f"Tensor Cores: {tensor_cores} (Expected: {expected_specs['Tensor Cores']})")
    print(f"Tensor Cores Match: {tensor_cores == expected_specs['Tensor Cores']}")

    # 6. RT Cores
    # Ada Lovelace: 1 RT Core per SM
    rt_cores_per_sm = 1  # Based on Ada Lovelace architecture
    rt_cores = sm_count * rt_cores_per_sm
    print(f"RT Cores: {rt_cores} (Expected: {expected_specs['RT Cores']})")
    print(f"RT Cores Match: {rt_cores == expected_specs['RT Cores']}")

    # 7. L1 Cache per SM
    # CUDA does not directly expose L1 cache size per SM. Use expected value.
    print(f"L1 Cache per SM: Cannot query directly via PyTorch/CUDA. Expected: {expected_specs['L1 Cache per SM']}")
    print("Note: L1 cache size requires NVIDIA documentation or profiling tools.")

    # 8. L2 Cache
    # Convert L2 cache to bytes for comparison
    expected_l2_cache_bytes = 32 * 1024 * 1024  # 32 MB in bytes
    l2_cache_bytes = props.l2_cache_size
    l2_cache_mb = l2_cache_bytes / (1024 * 1024)
    print(f"L2 Cache: {l2_cache_mb:.2f} MB (Expected: {expected_specs['L2 Cache']})")
    print(f"L2 Cache Match: {abs(l2_cache_mb - 32) < 0.1}")  # Allow small floating-point differences

if __name__ == "__main__":
    verify_gpu_specs()