# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# This file contains code originally written by Wenhao Li.
# --------------------------------------------------------

import math
import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    cu_seqlen,
    stride_oh,
    stride_om,
    stride_doh,
    stride_dom,
    seqlen_q,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m, off_b, off_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    m_begin = tl.load(cu_seqlen + off_b)
    m_end = tl.load(cu_seqlen + off_b + 1)

    if (start_m * BLOCK_M + BLOCK_M <= m_begin) or (start_m * BLOCK_M >= m_end):
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    mask_m = (offs_m >= m_begin) & (offs_m < m_end)

    o = tl.load(
        Out + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    do = tl.load(
        DO
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_h * seqlen_q + offs_m, delta, mask=mask_m)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.jit
def _fwd_kernel(
    Q, K, V,
    sink, cu_seqlen, sliding,
    Out, Lse,
    softmax_scale,
    stride_qh, stride_qm,
    stride_oh, stride_om,
    stride_kvh, stride_kvn,
    nheads,
    seqlen_q, 
    seqlen_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ENABLE_SLIDING: tl.constexpr,
):
    start_m_block, off_b, off_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    seq_beg = tl.load(cu_seqlen + off_b)
    seq_end = tl.load(cu_seqlen + off_b + 1)
    inner_batch_seqlen = seq_end - seq_beg

    if (start_m_block * BLOCK_M + BLOCK_M <= seq_beg) or (start_m_block * BLOCK_M >= seq_end):
        return

    off_kv_h = off_h // GROUP_SIZE

    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    mask_m = (offs_m >= seq_beg) & (offs_m < seq_end)
    q_ptrs = Q + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_kv_h * stride_kvh + (seq_beg + offs_n[:, None]) * stride_kvn + offs_d[None, :]
    v_ptrs = V + off_kv_h * stride_kvh + (seq_beg + offs_n[:, None]) * stride_kvn + offs_d[None, :]

    sink_logit = tl.load(sink + off_h)
    m_i = tl.full([BLOCK_M], sink_logit, dtype=tl.float32)
    lse_i = tl.full([BLOCK_M], sink_logit, dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    num_kv_blocks = max(0, tl.cdiv((start_m_block + 1) * BLOCK_M, BLOCK_N) - seq_beg // BLOCK_N)

    if ENABLE_SLIDING:
        for kv_block_idx in tl.range(0, num_kv_blocks):
            if (kv_block_idx * BLOCK_N + BLOCK_N + seq_beg + sliding) > (start_m_block * BLOCK_M):

                k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                kv_mask = k_idx[:, None] < inner_batch_seqlen

                k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
                v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
                qk = tl.dot(q, k.T)

                cond1 = offs_m[:, None] >= (k_idx + seq_beg)[None, :]
                cond2 = (k_idx + seq_beg + sliding)[None, :] > offs_m[:, None]
                qk = tl.where(cond1 & cond2, qk, float('-inf'))

                m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, m_i)
                p = tl.exp(qk * softmax_scale - m_ij[:, None])
                l_ij = tl.sum(p, 1)

                acc_o_scale = tl.exp(m_i - m_ij)
                acc_o = acc_o * acc_o_scale[:, None]
                p = p.to(v.dtype)
                acc_o += tl.dot(p, v)

                m_i = m_ij
                l_i_new = tl.exp(lse_i - m_ij) + l_ij
                lse_i = m_ij + tl.log(l_i_new)

            k_ptrs += BLOCK_N * stride_kvn
            v_ptrs += BLOCK_N * stride_kvn

    else:
        for kv_block_idx in tl.range(0, num_kv_blocks):

            k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            kv_mask = k_idx[:, None] < inner_batch_seqlen

            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
            qk = tl.dot(q, k.T)

            cond = offs_m[:, None] >= (seq_beg + k_idx)[None, :]
            qk = tl.where(cond, qk, float("-inf"))

            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, m_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])

            l_ij = tl.sum(p, 1)
            acc_o_scale = tl.exp(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]

            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

            k_ptrs += BLOCK_N * stride_kvn
            v_ptrs += BLOCK_N * stride_kvn

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]

    lse_ptrs = Lse + off_h * seqlen_q + offs_m
    tl.store(lse_ptrs, lse_i, mask=mask_m)

    out_ptrs = Out + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(out_ptrs, acc_o, mask=mask_m[:, None])


@triton.jit
def _bwd_kernel(
    Q, K, V, DO, DQ, DK, DV,
    sink, DSink, cu_seqlen, sliding, 
    LSE, D,
    softmax_scale,
    stride_qh, stride_qm,
    stride_kvh, stride_kvn,
    stride_doh, stride_dom,
    stride_dqh, stride_dqm,
    stride_dkvh, stride_dkvn,
    seqlen_q, 
    seqlen_k,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ENABLE_SLIDING: tl.constexpr,

):
    start_m_block, off_b, off_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    m_begin = tl.load(cu_seqlen + off_b)
    m_end = tl.load(cu_seqlen + off_b + 1)

    if (start_m_block * BLOCK_M + BLOCK_M <= m_begin) or (start_m_block * BLOCK_M >= m_end):
        return

    inner_batch_seqlen = m_end - m_begin

    off_kv_h = off_h // GROUP_SIZE
    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    do_ptrs = DO + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :]
    dq_ptrs = DQ + off_h * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :]

    k_ptrs = K + off_kv_h * stride_kvh + (m_begin + offs_n)[:, None] * stride_kvn + offs_d[None, :]
    v_ptrs = V + off_kv_h * stride_kvh + (m_begin + offs_n)[:, None] * stride_kvn + offs_d[None, :]
    dk_ptrs = DK + off_kv_h * stride_dkvh + (m_begin + offs_n)[:, None] * stride_dkvn + offs_d[None, :]
    dv_ptrs = DV + off_kv_h * stride_dkvh + (m_begin + offs_n)[:, None] * stride_dkvn + offs_d[None, :]

    dsink_ptr = DSink + off_h
    lse_ptrs = LSE + off_h * seqlen_q + offs_m
    d_ptrs = D + off_h * seqlen_q + offs_m
    mask_m = (offs_m >= m_begin) and (offs_m < m_end)

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    lse_i = tl.load(lse_ptrs, mask=mask_m, other=0.0)
    Di = tl.load(d_ptrs, mask=mask_m, other=0.0)

    sink_logit = tl.load(sink + off_h)
    p_sink = tl.exp(sink_logit - lse_i)

    d_sink_per_query = -p_sink * Di
    d_sink_per_query = tl.where(mask_m, d_sink_per_query, 0.0)
    d_sink_total = tl.sum(d_sink_per_query, axis=0)

    tl.atomic_add(dsink_ptr, d_sink_total, sem='relaxed')
    dq_block = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    num_kv_blocks = max(0, tl.cdiv((start_m_block + 1) * BLOCK_M, BLOCK_N) - m_begin // BLOCK_N)

    if ENABLE_SLIDING:
        for kv_block_idx in tl.range(0, num_kv_blocks):
            if (kv_block_idx * BLOCK_N + BLOCK_N + m_begin + sliding) > (start_m_block * BLOCK_M):

                k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                kv_mask = k_idx[:, None] < inner_batch_seqlen

                k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
                v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
                qk = tl.dot(q, k.T)

                cond1 = offs_m[:, None] >= (k_idx + m_begin)[None, :]
                cond2 = (k_idx + m_begin + sliding)[None, :] > offs_m[:, None]
                qk = tl.where(cond1 & cond2, qk, float('-inf'))
                p = tl.exp(qk * softmax_scale - lse_i[:, None])

                dv_block = tl.dot(p.to(do.dtype).T, do)
                tl.atomic_add(dv_ptrs, dv_block, mask=kv_mask, sem='relaxed')

                dp = tl.dot(do, v.T)
                ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

                dk_block = tl.dot(ds.T, q)
                tl.atomic_add(dk_ptrs, dk_block, mask=kv_mask, sem='relaxed')

                dq_block += tl.dot(ds, k)

            k_ptrs += BLOCK_N * stride_kvn
            v_ptrs += BLOCK_N * stride_kvn
            dk_ptrs += BLOCK_N * stride_dkvn
            dv_ptrs += BLOCK_N * stride_dkvn

    else:
        for kv_block_idx in tl.range(0, num_kv_blocks):

            k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            kv_mask = k_idx[:, None] < inner_batch_seqlen

            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
            qk = tl.dot(q, k.T)

            cond = offs_m[:, None] >= (k_idx + m_begin)[None, :]
            qk = tl.where(cond, qk, float('-inf'))
            p = tl.exp(qk * softmax_scale - lse_i[:, None])

            dv_block = tl.dot(p.to(do.dtype).T, do)
            tl.atomic_add(dv_ptrs, dv_block, mask=kv_mask, sem='relaxed')

            dp = tl.dot(do, v.T)
            ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

            dk_block = tl.dot(ds.T, q)
            tl.atomic_add(dk_ptrs, dk_block, mask=kv_mask, sem='relaxed')

            dq_block += tl.dot(ds, k)
            k_ptrs += BLOCK_N * stride_kvn
            v_ptrs += BLOCK_N * stride_kvn
            dk_ptrs += BLOCK_N * stride_dkvn
            dv_ptrs += BLOCK_N * stride_dkvn

    tl.store(dq_ptrs, dq_block, mask=mask_m[:, None])


def _flash_attn_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: torch.Tensor, 
        cu_seqlen: torch.Tensor,
        sliding: int,
        softmax_scale=None):

    global BLOCK_M, BLOCK_N
    seqlen_q, nheads, d = q.shape
    seqlen_k, num_kv_heads = k.shape[:2]

    assert d <= 128
    assert q.is_cuda

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    lse = torch.empty((nheads, seqlen_q), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    GROUP_SIZE = nheads // num_kv_heads

    grid = (triton.cdiv(seqlen_q, BLOCK_M), cu_seqlen.numel() - 1, nheads)
    num_warps = 4 if d <= 64 else 8

    _fwd_kernel[grid](
        q, k, v,
        sink, cu_seqlen, sliding,
        o, lse,
        softmax_scale,
        q.stride(1), q.stride(0),
        o.stride(1), o.stride(0),
        k.stride(1), k.stride(0),
        nheads, 
        seqlen_q,
        seqlen_k, 
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=(seqlen_q % BLOCK_M == 0),
        GROUP_SIZE=GROUP_SIZE,
        ENABLE_SLIDING=(sliding is not None),
        num_warps=num_warps,
        num_stages=1)

    return o, lse, softmax_scale


def _flash_attn_backward(
        o: torch.Tensor,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        sink: torch.Tensor,
        dsink: torch.Tensor,
        cu_seqlen: torch.Tensor,
        sliding: int,
        lse: torch.Tensor,
        softmax_scale: float
):
    if do.stride(-1) != 1:
        do = do.contiguous()

    global BLOCK_M, BLOCK_N
    seqlen_q, nheads, d = q.shape
    seqlen_k, num_kv_heads, kv_head_dim = k.shape
    delta = torch.empty_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    GROUP_SIZE = nheads // num_kv_heads
    grid_preprocess = (triton.cdiv(seqlen_q, BLOCK_M), cu_seqlen.numel() - 1, nheads)

    _bwd_preprocess_do_o_dot[grid_preprocess](
        o, do, delta,
        cu_seqlen,
        o.stride(1), 
        o.stride(0),
        do.stride(1), 
        do.stride(0),
        seqlen_q,
        BLOCK_M=BLOCK_M, 
        BLOCK_HEADDIM=BLOCK_HEADDIM)

    num_warps = 4 if kv_head_dim <= 64 else 8
    grid_bwd = (triton.cdiv(seqlen_q, BLOCK_M), cu_seqlen.numel() - 1, nheads)

    _bwd_kernel[grid_bwd](
        q, k, v, do, dq, dk, dv,
        sink, dsink, cu_seqlen, sliding,
        lse, delta,
        softmax_scale,
        q.stride(1), q.stride(0),
        k.stride(1), k.stride(0),
        do.stride(1), do.stride(0),
        dq.stride(1), dq.stride(0),
        dk.stride(1), dk.stride(0),
        seqlen_q, 
        seqlen_k, 
        BLOCK_M=BLOCK_M, 
        BLOCK_N=BLOCK_N,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        GROUP_SIZE=GROUP_SIZE,
        ENABLE_SLIDING=(sliding is not None),
        num_warps=num_warps,
        num_stages=1)


class FlashSinkVarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            sink: torch.Tensor,
            cu_seqlen: torch.Tensor,
            manager):

        q = q if q.stride(-1) == 1 else q.contiguous()

        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, manager.key, manager.val,
            sink=sink,
            cu_seqlen=cu_seqlen,
            sliding=manager.sliding_window,
            softmax_scale=None)

        ctx.save_for_backward(q, o, lse)
        ctx.sink = sink
        ctx.manager = manager
        ctx.cu_seqlen = cu_seqlen

        return o

    @staticmethod
    def backward(ctx, do):
        q, o, lse = ctx.saved_tensors

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(ctx.manager.key)
        dv = torch.zeros_like(ctx.manager.val)
        dsink = torch.zeros_like(ctx.sink)

        _flash_attn_backward(
            o, do, q, ctx.manager.key, ctx.manager.val, dq, dk, dv,
            ctx.sink, 
            dsink,
            ctx.cu_seqlen,
            ctx.manager.sliding_window,
            lse,
            ctx.softmax_scale)

        return dq, dk, dv, dsink, None, None

flash_sink_attn_varlen_func = FlashSinkVarlenAttention.apply
