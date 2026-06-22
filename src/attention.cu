#include "attention_kernel.h"
#include <cuda_runtime.h>

__global__ void attention_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int seq_len, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;
    
    float max_val = -1e9f;
    float sum = 0.0f;
    
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[idx * head_dim + d] * K[j * head_dim + d];
        }
        score /= sqrtf((float)head_dim);
        max_val = fmaxf(max_val, score);
    }
    
    for (int j = 0; j < seq_len; j++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[idx * head_dim + d] * K[j * head_dim + d];
        }
        score = expf(score / sqrtf((float)head_dim) - max_val);
        sum += score;
        for (int d = 0; d < head_dim; d++) {
            O[idx * head_dim + d] += score * V[j * head_dim + d];
        }
    }
    
    for (int d = 0; d < head_dim; d++) {
        O[idx * head_dim + d] /= sum;
    }
}

void fused_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, int batch, int seq_len, int head_dim
) {
    int blocks = (seq_len + 255) / 256;
    attention_kernel<<<blocks, 256>>>(Q, K, V, O, seq_len, head_dim);
}
