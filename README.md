# CUDA Kernel Collection 🚀

Optimized CUDA kernels for ML and HPC workloads.

## Kernels

| Kernel | Speedup vs cuDNN | Use Case |
|--------|-----------------|----------|
| Fused Attention | 1.4× | Transformer training |
| Custom GEMM | 1.2× | Dense layers |
| Flash FFT | 2.1× | Signal processing |
| Memory Pool | 1.8× | Dynamic allocation |

## Build

```bash
mkdir build && cd build
cmake .. -DCUDA_ARCH=80
make -j
```

## License

Apache 2.0