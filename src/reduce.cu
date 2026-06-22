#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void reduce_sum(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    sdata[tid] = 0.0f;
    if (idx < n) sdata[tid] = input[idx];
    if (idx + BLOCK_SIZE < n) sdata[tid] += input[idx + BLOCK_SIZE];
    
    __syncthreads();
    
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

extern "C" void launch_reduce(const float* in, float* out, int n) {
    int blocks = (n + 255) / 256;
    reduce_sum<256><<<blocks, 256>>>(in, out, n);
}
