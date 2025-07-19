"""
Optimized CUDA kernels for Dendritic Attention using Triton
Designed for RTX 3090's Ampere architecture
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def dendritic_attention_fwd_kernel(
    Q, K, V, Out,
    Importance, 
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_ib, stride_im,
    nheads, seqlen, d_head,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IMPORTANCE_THRESH: tl.constexpr,
):
    """
    Fused dendritic attention kernel with importance gating
    Processes only important tokens based on gating threshold
    """
    # Get program ids
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # head 
    pid_m = tl.program_id(2)  # sequence position
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Load importance scores for current tokens
    importance_ptrs = Importance + pid_b * stride_ib + offs_m * stride_im
    importance = tl.load(importance_ptrs, mask=offs_m < seqlen, other=0.0)
    
    # Skip computation if importance is below threshold
    if tl.sum(importance > IMPORTANCE_THRESH) == 0:
        # Write zeros for unimportant tokens
        out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(out_ptrs, tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32), mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < d_head))
        return
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    # Load Q
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < d_head), other=0.0)
    
    # Apply importance gating to Q
    q = q * importance[:, None]
    
    # Causal attention loop
    for n in range(0, seqlen, BLOCK_N):
        # Load K and V blocks
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + (n + offs_n[None, :]) * stride_kn + offs_k[:, None] * stride_kk
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + (n + offs_n[None, :]) * stride_vn + offs_k[:, None] * stride_vk
        
        # Causal mask
        mask = (offs_m[:, None] >= (n + offs_n[None, :])) & ((n + offs_n[None, :]) < seqlen)
        
        k = tl.load(k_ptrs, mask=mask & (offs_k[:, None] < d_head), other=0.0)
        v = tl.load(v_ptrs, mask=mask & (offs_k[:, None] < d_head), other=0.0)
        
        # Compute attention scores
        scores = tl.dot(q, k) / math.sqrt(d_head)
        
        # Apply causal mask
        scores = tl.where(mask, scores, float('-inf'))
        
        # Softmax
        scores_max = tl.max(scores, axis=1)[:, None]
        scores = scores - scores_max
        scores_exp = tl.exp(scores)
        scores_sum = tl.sum(scores_exp, axis=1)[:, None] + 1e-6
        scores_norm = scores_exp / scores_sum
        
        # Accumulate weighted values
        acc += tl.dot(scores_norm, v.trans())
    
    # Store output
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < seqlen) & (offs_k[None, :] < d_head))


@triton.jit
def cortical_column_sparse_matmul_kernel(
    A, B, C, 
    Mask,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn, 
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Optimized sparse matrix multiplication for cortical column patterns
    Uses 2:4 structured sparsity for RTX 3090 tensor cores
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Number of blocks
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    
    # Swizzle for better L2 cache utilization
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop with 2:4 sparsity pattern
    for k in range(0, K, BLOCK_K):
        # Load A block
        a_ptrs = A + offs_am[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        
        # Load B block
        b_ptrs = B + (k + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_bn[None, :] < N), other=0.0)
        
        # Apply 2:4 structured sparsity mask
        # Keep only 2 out of every 4 elements for Ampere optimization
        if k % 4 == 0:
            # Create 2:4 mask pattern
            mask_k = (offs_k % 4) < 2
            a = a * mask_k[None, :]
            b = b * mask_k[:, None]
        
        # Matrix multiplication
        acc += tl.dot(a, b)
    
    # Store result
    c_ptrs = C + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))


@triton.jit
def lateral_inhibition_kernel(
    Input, Output, 
    Inhibition_weights,
    batch_size, seq_len, n_columns, column_size,
    stride_ib, stride_is, stride_ic, stride_in,
    stride_ob, stride_os, stride_oc, stride_on,
    BLOCK_C: tl.constexpr,
):
    """
    Implements lateral inhibition between cortical columns
    Creates winner-take-all dynamics within local neighborhoods
    """
    # Program IDs
    pid_b = tl.program_id(0)  # batch
    pid_s = tl.program_id(1)  # sequence position
    pid_c = tl.program_id(2)  # column
    
    # Column offsets
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_n = tl.arange(0, column_size)
    
    # Load current column data
    input_ptrs = Input + pid_b * stride_ib + pid_s * stride_is + offs_c[:, None] * stride_ic + offs_n[None, :] * stride_in
    col_data = tl.load(input_ptrs, mask=(offs_c[:, None] < n_columns) & (offs_n[None, :] < column_size), other=0.0)
    
    # Compute column activities (mean activation)
    col_activity = tl.mean(col_data, axis=1)
    
    # Load inhibition weights for current columns
    inhibition_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    for other_c in range(0, n_columns, BLOCK_C):
        # Load other column activities
        other_offs = other_c + tl.arange(0, BLOCK_C)
        other_input_ptrs = Input + pid_b * stride_ib + pid_s * stride_is + other_offs[:, None] * stride_ic + offs_n[None, :] * stride_in
        other_data = tl.load(other_input_ptrs, mask=(other_offs[:, None] < n_columns) & (offs_n[None, :] < column_size), other=0.0)
        other_activity = tl.mean(other_data, axis=1)
        
        # Load inhibition weights
        inh_weight_ptrs = Inhibition_weights + offs_c[:, None] * n_columns + other_offs[None, :]
        inh_weights = tl.load(inh_weight_ptrs, mask=(offs_c[:, None] < n_columns) & (other_offs[None, :] < n_columns), other=0.0)
        
        # Accumulate inhibition
        inhibition_sum += tl.sum(inh_weights * other_activity[None, :], axis=1)
    
    # Apply inhibition with ReLU
    inhibited_data = col_data - inhibition_sum[:, None] * col_data
    inhibited_data = tl.maximum(inhibited_data, 0.0)
    
    # Store output
    output_ptrs = Output + pid_b * stride_ob + pid_s * stride_os + offs_c[:, None] * stride_oc + offs_n[None, :] * stride_on
    tl.store(output_ptrs, inhibited_data, mask=(offs_c[:, None] < n_columns) & (offs_n[None, :] < column_size))


class DendriticAttentionCUDA(torch.nn.Module):
    """
    CUDA-optimized dendritic attention implementation
    """
    
    def __init__(self, n_heads, d_head, importance_threshold=0.3):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.importance_threshold = importance_threshold
        
    def forward(self, q, k, v, importance_scores):
        # Ensure contiguous tensors
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        importance_scores = importance_scores.contiguous()
        
        batch_size, n_heads, seq_len, d_head = q.shape
        
        # Allocate output
        out = torch.empty_like(q)
        
        # Grid and block sizes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        
        grid = (batch_size, n_heads, triton.cdiv(seq_len, BLOCK_M))
        
        # Launch kernel
        dendritic_attention_fwd_kernel[grid](
            q, k, v, out,
            importance_scores,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            importance_scores.stride(0), importance_scores.stride(1),
            n_heads, seq_len, d_head,
            BLOCK_M, BLOCK_N, BLOCK_K,
            self.importance_threshold,
        )
        
        return out


class CorticalColumnSparseMM(torch.autograd.Function):
    """
    Custom autograd function for cortical column sparse matrix multiplication
    """
    
    @staticmethod
    def forward(ctx, a, b, mask=None):
        # Ensure contiguous
        a = a.contiguous()
        b = b.contiguous()
        
        M, K = a.shape
        K_b, N = b.shape
        assert K == K_b, "Matrix dimensions must match"
        
        # Allocate output
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        
        # Configure kernel
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        GROUP_M = 8
        
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        # Launch kernel
        cortical_column_sparse_matmul_kernel[grid](
            a, b, c, mask,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
        )
        
        ctx.save_for_backward(a, b)
        return c
        
    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        
        # Gradient w.r.t. a: grad_c @ b.T
        grad_a = CorticalColumnSparseMM.apply(grad_c, b.T)
        
        # Gradient w.r.t. b: a.T @ grad_c
        grad_b = CorticalColumnSparseMM.apply(a.T, grad_c)
        
        return grad_a, grad_b, None


# Convenience function
cortical_sparse_mm = CorticalColumnSparseMM.apply