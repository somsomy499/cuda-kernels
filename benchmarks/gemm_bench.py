"""GEMM benchmark comparing naive vs tiled."""
import numpy as np
import time

def bench_gemm(M, K, N, iterations=100):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    start = time.time()
    for _ in range(iterations):
        C = A @ B
    elapsed = (time.time() - start) / iterations
    
    flops = 2 * M * N * K
    tflops = flops / elapsed / 1e12
    
    print(f"GEMM {M}x{K}x{N}: {elapsed*1000:.2f}ms, {tflops:.2f} TFLOPS")
    return elapsed

if __name__ == "__main__":
    for size in [256, 512, 1024, 2048]:
        bench_gemm(size, size, size)
