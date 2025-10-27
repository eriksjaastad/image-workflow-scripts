#!/usr/bin/env python3
"""
MPS Benchmark Test for Apple M4 Pro
Tests PyTorch Metal Performance Shaders (MPS) acceleration

This is a standalone benchmark script, not a pytest test.
Run directly: python scripts/tests/test_mps_benchmark.py
"""

import platform
import sys
import time

# Not a pytest test - skip collection
__test__ = False

def test_pytorch_installation():
    """Test if PyTorch is installed"""
    print("=" * 70)
    print("PyTorch MPS Benchmark Test - Apple M4 Pro")
    print("=" * 70)
    print()
    
    try:
        import torch
        print(f"✅ PyTorch installed: v{torch.__version__}")
        return torch
    except ImportError:
        print("❌ PyTorch not installed!")
        print("\nInstall with:")
        print("  pip install torch torchvision torchaudio")
        sys.exit(1)

def test_mps_availability(torch):
    """Test if MPS backend is available"""
    print(f"\n{'=' * 70}")
    print("MPS Device Availability")
    print("=" * 70)
    
    # System info
    print(f"\nSystem: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # MPS availability
    if torch.backends.mps.is_available():
        print("\n✅ MPS backend is AVAILABLE")
        
        if torch.backends.mps.is_built():
            print("✅ MPS backend is BUILT")
        else:
            print("⚠️  MPS backend not built (unexpected)")
            
        return True
    else:
        print("\n❌ MPS backend is NOT AVAILABLE")
        print("\nPossible reasons:")
        print("  - Not running on Apple Silicon Mac")
        print("  - macOS < 12.3")
        print("  - PyTorch not built with MPS support")
        return False

def benchmark_cpu_vs_mps(torch):
    """Benchmark CPU vs MPS performance"""
    print(f"\n{'=' * 70}")
    print("Performance Benchmark: CPU vs MPS")
    print("=" * 70)
    
    # Test configuration
    matrix_size = 4096
    iterations = 20
    
    print(f"\nTest: Matrix multiplication ({matrix_size}x{matrix_size})")
    print(f"Iterations: {iterations}")
    
    # CPU Benchmark
    print("\n--- CPU Benchmark ---")
    device_cpu = torch.device("cpu")
    
    # Create random matrices on CPU
    a_cpu = torch.randn(matrix_size, matrix_size, device=device_cpu)
    b_cpu = torch.randn(matrix_size, matrix_size, device=device_cpu)
    
    # Warmup
    _ = torch.matmul(a_cpu, b_cpu)
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    
    print(f"CPU Time: {cpu_time:.3f}s ({cpu_time/iterations:.4f}s per iteration)")
    
    # MPS Benchmark
    if torch.backends.mps.is_available():
        print("\n--- MPS Benchmark ---")
        device_mps = torch.device("mps")
        
        # Create random matrices on MPS
        a_mps = torch.randn(matrix_size, matrix_size, device=device_mps)
        b_mps = torch.randn(matrix_size, matrix_size, device=device_mps)
        
        # Warmup
        _ = torch.matmul(a_mps, b_mps)
        torch.mps.synchronize()  # Wait for GPU to finish
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            _ = torch.matmul(a_mps, b_mps)
        torch.mps.synchronize()  # Wait for GPU to finish
        mps_time = time.time() - start
        
        print(f"MPS Time: {mps_time:.3f}s ({mps_time/iterations:.4f}s per iteration)")
        
        # Speedup
        speedup = cpu_time / mps_time
        print(f"\n🚀 Speedup: {speedup:.2f}x faster on MPS")
        
        return {"cpu": cpu_time, "mps": mps_time, "speedup": speedup}
    else:
        print("\n⚠️  Skipping MPS benchmark (not available)")
        return {"cpu": cpu_time, "mps": None, "speedup": None}

def test_embedding_simulation(torch):
    """Simulate CLIP-style embedding extraction"""
    print(f"\n{'=' * 70}")
    print("Embedding Extraction Simulation (CLIP-style)")
    print("=" * 70)
    
    batch_size = 32
    embedding_dim = 512
    num_batches = 10
    
    print(f"\nSimulating: {batch_size} images/batch, {embedding_dim}D embeddings")
    print(f"Batches: {num_batches}")
    
    # Simulate feature extraction (like CLIP ViT-B/32)
    # Typical CLIP: ~224x224 image → encoder → 512D embedding
    
    for device_name, device in [("CPU", torch.device("cpu")), 
                                 ("MPS", torch.device("mps") if torch.backends.mps.is_available() else None)]:
        if device is None:
            continue
            
        print(f"\n--- {device_name} ---")
        
        # Simulate a simple encoder (linear layers)
        model = torch.nn.Sequential(
            torch.nn.Linear(224*224*3, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, embedding_dim)
        ).to(device)
        
        # Simulate image batch
        images = torch.randn(batch_size, 224*224*3, device=device)
        
        # Warmup
        with torch.no_grad():
            _ = model(images)
        if device.type == "mps":
            torch.mps.synchronize()
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(images)
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - start
        
        total_images = batch_size * num_batches
        imgs_per_sec = total_images / elapsed
        
        print(f"Time: {elapsed:.3f}s")
        print(f"Throughput: {imgs_per_sec:.1f} images/second")
        print(f"Per image: {1000*elapsed/total_images:.2f}ms")

def test_memory_info(torch):
    """Display memory information"""
    print(f"\n{'=' * 70}")
    print("Memory Information")
    print("=" * 70)
    
    if torch.backends.mps.is_available():
        print("\n⚠️  Note: PyTorch MPS does not yet expose memory stats")
        print("    (unlike CUDA's torch.cuda.memory_allocated())")
        print("    Memory is managed automatically by Metal")
    else:
        print("\nMPS not available - using CPU only")

def main():
    """Run all benchmarks"""
    torch = test_pytorch_installation()
    mps_available = test_mps_availability(torch)
    
    if mps_available:
        benchmark_cpu_vs_mps(torch)
        test_embedding_simulation(torch)
        test_memory_info(torch)
    else:
        print("\n" + "=" * 70)
        print("⚠️  MPS not available - AI training will use CPU only")
        print("=" * 70)
        print("\nThis is okay but will be slower for:")
        print("  - Embedding extraction (OpenCLIP)")
        print("  - Model training (ranking, crop proposer)")
        print("\nMost preprocessing can still run efficiently on CPU:")
        print("  - MediaPipe Hands")
        print("  - pHash computation")
        print("  - U²-Net saliency (small models)")
    
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    print("\nRecommendations for AI training:")
    if mps_available:
        print("  ✅ Use device='mps' for:")
        print("     - OpenCLIP embedding extraction")
        print("     - Model training (Bradley-Terry, MLP)")
        print("  ✅ Use device='cpu' for:")
        print("     - MediaPipe Hands")
        print("     - pHash computation")
        print("     - U²-Net saliency (if issues with MPS)")
    else:
        print("  ✅ Use device='cpu' for all operations")
        print("  ⚠️  Consider installing PyTorch with MPS support")
    print()

if __name__ == "__main__":
    main()
