#pragma once
void fused_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, int batch, int seq_len, int head_dim
);
