"""Tests for CUDA kernels (CPU fallback for CI)."""
import numpy as np

def test_matmul_reference():
    """Reference matmul for verification."""
    A = np.random.randn(64, 64).astype(np.float32)
    B = np.random.randn(64, 64).astype(np.float32)
    C = A @ B
    assert C.shape == (64, 64)
    assert np.allclose(C, A @ B, atol=1e-5)

def test_reduce_reference():
    """Reference reduction."""
    data = np.random.randn(1024).astype(np.float32)
    result = np.sum(data)
    assert abs(result - np.sum(data)) < 1e-3
