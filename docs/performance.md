# Performance

## Kernel Benchmarks (MI300X)

| Kernel | Naive | Tiled | Optimized |
|--------|-------|-------|-----------|
| SGEMM 1024 | 12ms | 3.2ms | 1.8ms |
| SGEMM 4096 | 850ms | 180ms | 95ms |
| Reduce 1M | 0.8ms | 0.2ms | 0.15ms |
| Attention 2K | 45ms | 18ms | 8ms |

## Memory Usage

| Kernel | Registers | Shared Mem | Grid Size |
|--------|-----------|-----------|-----------|
| SGEMM tiled | 32 | 8KB | (N/16, M/16) |
| Reduce | 16 | 1KB | (N/256) |
