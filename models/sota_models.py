#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State-of-the-Art (SOTA) Models for HGV Trajectory Prediction

This module contains implementations of the latest SOTA models for trajectory prediction,
including advanced Transformer variants and physics-informed neural networks.

包含以下SOTA模型：
1. Informer - ProbSparse注意力机制 (AAAI 2021)
2. PatchTST - 基于Patch的时序Transformer (ICLR 2023)
3. FEDformer - 频域增强Transformer (ICML 2022)
4. TimesNet - 基于时序2D卷积的Transformer (ICLR 2023)
5. iTransformer - 倒置Transformer (NeurIPS 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any


class PositionalEncoding(nn.Module):
    """标准位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # 标准正弦位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x, phase_info=None):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)


# ======================================================================================
# Autoformer - decomposition and auto-correlation blocks (NeurIPS 2021)
# ======================================================================================

class SeriesDecomposition(nn.Module):
    """Moving-average decomposition used by Autoformer-style encoders."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kernel_size <= 1:
            trend = x
            return x - trend, trend
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, :1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(padded.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelationAttention(nn.Module):
    """FFT-based time-delay aggregation inspired by Autoformer."""

    def __init__(self, d_model: int, n_heads: int, factor: int = 3, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = max(1, int(factor))
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q_fft = torch.fft.rfft(q.float(), dim=2)
        k_fft = torch.fft.rfft(k.float(), dim=2)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=seq_len, dim=2).mean(dim=-1)
        corr = corr.to(dtype=x.dtype)
        top_k = max(1, min(seq_len, self.factor * int(math.log(max(seq_len, 2)))))
        values, delays = torch.topk(corr, k=top_k, dim=-1)
        weights = torch.softmax(values, dim=-1)

        base = torch.arange(seq_len, device=x.device).view(1, 1, 1, seq_len)
        gather_idx = (base + delays.unsqueeze(-1)) % seq_len
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        shifted = torch.gather(
            v.unsqueeze(2).expand(-1, -1, top_k, -1, -1),
            dim=3,
            index=gather_idx,
        )
        context = (weights.unsqueeze(-1).unsqueeze(-1) * shifted).sum(dim=2)

        context = self.dropout(context)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class AutoformerEncoderLayer(nn.Module):
    """Autoformer encoder layer with progressive decomposition."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, moving_avg: int = 25, factor: int = 3, dropout: float = 0.1):
        super().__init__()
        self.auto_corr = AutoCorrelationAttention(d_model, n_heads, factor=factor, dropout=dropout)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.auto_corr(x))
        seasonal, trend1 = self.decomp1(x)
        seasonal = seasonal + self.ff(seasonal)
        seasonal, trend2 = self.decomp2(seasonal)
        return self.norm(seasonal), trend1 + trend2


class Autoformer(nn.Module):
    """Autoformer baseline with decomposition and auto-correlation aggregation."""

    def __init__(self, input_dim=6, d_model=None, n_heads=None, e_layers=None, d_layers=None,
                 d_ff=None, dropout=None, moving_avg=25, factor=3, seq_len=None, pred_len=None):
        super().__init__()
        from . import HGVConfig
        config = HGVConfig.get_model_config('autoformer')
        train_config = HGVConfig.get_train_config()
        self.input_dim = input_dim
        self.output_dim = 3
        self.d_model = d_model or config['d_model']
        self.n_heads = n_heads or config['nhead']
        self.e_layers = e_layers or config['num_encoder_layers']
        self.d_ff = d_ff or config['dim_feedforward']
        self.dropout = dropout or config['dropout']
        self.seq_len = seq_len or train_config.get('seq_len', 64)
        self.pred_len = pred_len or train_config.get('pred_len', 128)

        self.enc_embedding = nn.Linear(input_dim, self.d_model)
        self.context_embedding = nn.Linear(self.output_dim, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)
        self.decomposition = SeriesDecomposition(moving_avg)
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(
                self.d_model,
                self.n_heads,
                self.d_ff,
                moving_avg=moving_avg,
                factor=factor,
                dropout=self.dropout,
            )
            for _ in range(self.e_layers)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.pred_len * self.output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        target_length: int = None,
        decoder_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        enc_out = self.pos_encoding(self.enc_embedding(x))
        seasonal, trend = self.decomposition(enc_out)
        trend_acc = trend

        for layer in self.encoder_layers:
            seasonal, trend_res = layer(seasonal)
            trend_acc = trend_acc + trend_res

        pooled = seasonal.mean(dim=1) + trend_acc.mean(dim=1)
        if decoder_context is not None and decoder_context.numel() > 0:
            context = self.context_embedding(decoder_context[..., : self.output_dim]).mean(dim=1)
            pooled = pooled + 0.1 * context

        output = self.output_head(pooled).view(batch_size, self.pred_len, self.output_dim)
        if target_length is not None and target_length != self.pred_len:
            output = output.transpose(1, 2)
            output = F.interpolate(output, size=target_length, mode='linear', align_corners=False)
            output = output.transpose(1, 2)
        return output


# ======================================================================================
# Informer - ProbSparse注意力机制 (AAAI 2021)
# ======================================================================================

class InformerAttention(nn.Module):
    """Informer的ProbSparse注意力机制"""
    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = 1. / math.sqrt(d_model // n_heads)
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        """计算ProbSparse注意力的Q-K相关性"""
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        # 确保sample_k不超过L_K
        sample_k = min(sample_k, L_K)
        # 确保n_top不超过L_Q
        n_top = min(n_top, L_Q)
        
        # 计算采样的K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        
        # 计算Q-K相关性
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # 找到Top-k查询
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), sample_k)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # 使用reduced Q计算Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # 确保维度正确
        queries = self.query_projection(queries).view(B, L, H, self.head_dim)
        keys = self.key_projection(keys).view(B, S, H, self.head_dim)
        values = self.value_projection(values).view(B, S, H, self.head_dim)
        
        queries = queries.transpose(2, 1)  # [B, H, L, head_dim]
        keys = keys.transpose(2, 1)        # [B, H, S, head_dim]
        values = values.transpose(2, 1)    # [B, H, S, head_dim]
        
        # 计算采样参数，确保不超过序列长度
        U_part = max(1, min(L, self.factor * int(np.ceil(np.log(max(L, 1))))))
        u = max(1, min(S, self.factor * int(np.ceil(np.log(max(S, 1))))))
        
        # 简化的ProbSparse注意力计算
        if L > S:
            # 当查询序列较长时，使用最后S个查询
            queries_tmp = queries[:, :, -S:, :]
            scores_top, index = self._prob_QK(queries_tmp, keys, sample_k=U_part, n_top=u)
            
            # 添加尺度因子
            scores_top = scores_top * self.scale
            
            # 应用掩码
            if attn_mask is not None:
                # 调整掩码尺寸以匹配scores_top
                mask_size = scores_top.shape[-1]
                if attn_mask.shape[-1] != mask_size:
                    attn_mask = attn_mask[:, :, :, -mask_size:]
                if attn_mask.dtype == torch.bool:
                    scores_top.masked_fill_(attn_mask, -np.inf)
                else:
                    scores_top += attn_mask
                    
            A = torch.softmax(scores_top, dim=-1)
            V = values[:, :, -S:, :]
            
            # 计算注意力输出
            context = torch.matmul(A, V)  # [B, H, u, head_dim]
            
            # 扩展到完整长度
            full_context = torch.zeros(B, H, L, self.head_dim, device=context.device, dtype=context.dtype)
            full_context[:, :, -S:, :] = context
            context = full_context
        else:
            scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
            
            # 添加尺度因子
            scores_top = scores_top * self.scale
            
            # 应用掩码
            if attn_mask is not None:
                # 调整掩码尺寸以匹配scores_top
                mask_size = scores_top.shape[-1]
                if attn_mask.shape[-1] != mask_size:
                    attn_mask = attn_mask[:, :, :mask_size, :mask_size]
                if attn_mask.dtype == torch.bool:
                    scores_top.masked_fill_(attn_mask, -np.inf)
                else:
                    scores_top += attn_mask
                    
            A = torch.softmax(scores_top, dim=-1)
            
            # 计算注意力输出
            context = torch.matmul(A, values)  # [B, H, u, head_dim]
            
            # 扩展到完整长度
            full_context = torch.zeros(B, H, L, self.head_dim, device=context.device, dtype=context.dtype)
            full_context[:, :, :u, :] = context
            context = full_context
            
        context = context.transpose(2, 1).contiguous()  # [B, L, H, head_dim]
        context = context.view(B, L, self.d_model)     # [B, L, d_model]
        
        return self.out_projection(context)


class InformerEncoderLayer(nn.Module):
    """Informer编码器层"""
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, factor=5):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = InformerAttention(d_model, n_heads, factor, dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        new_x = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        y = x.transpose(-1, 1)
        y = self.dropout(F.relu(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(-1, 1)
        
        return self.norm2(x + y)


class Informer(nn.Module):
    """Informer模型实现"""
    def __init__(self, input_dim=6, d_model=None, n_heads=None, e_layers=None, d_layers=None, 
                 d_ff=None, factor=5, dropout=None):
        super(Informer, self).__init__()
        
        # 动态导入配置以避免循环导入
        from . import HGVConfig
        config = HGVConfig.get_model_config('informer')
        self.d_model = d_model or config['d_model']
        self.n_heads = n_heads or config['nhead']
        self.e_layers = e_layers or config['num_encoder_layers']
        self.d_layers = d_layers or config['num_decoder_layers']
        self.d_ff = d_ff or config['dim_feedforward']
        self.dropout = dropout or config['dropout']
        
        # 输入嵌入
        self.enc_embedding = nn.Linear(input_dim, self.d_model)
        self.dec_embedding = nn.Linear(3, self.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout, factor)
            for _ in range(self.e_layers)
        ])
        
        # 解码器 (简化为标准Transformer解码器)
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_model, self.n_heads, self.d_ff, self.dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, self.d_layers)
        
        # 输出投影
        self.projection = nn.Linear(self.d_model, 3)
        
    def forward(
        self,
        x: torch.Tensor,
        target_length: int = None,
        decoder_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            target_length: Target sequence length (if None, use config pred_len)
        Returns:
            Output tensor of shape (batch_size, target_length, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 确定输出长度
        if target_length is None:
            # 动态导入配置以获取pred_len
            from . import HGVConfig
            train_config = HGVConfig.get_train_config()
            pred_len = train_config['pred_len']  # 使用配置的预测长度
        else:
            pred_len = target_length
        
        # 编码器
        enc_out = self.enc_embedding(x)
        enc_out = self.pos_encoding(enc_out)
        
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        
        # 全局池化以获得固定长度的表示
        # 使用平均池化将seq_len维度压缩为1
        pooled = enc_out.mean(dim=1, keepdim=True)  # (batch_size, 1, d_model)
        
        # 可选 decoder context（official-like 协议）融合
        if decoder_context is not None and decoder_context.numel() > 0:
            dec_ctx = decoder_context[:, : min(decoder_context.size(1), pred_len), :]
            dec_emb = self.dec_embedding(dec_ctx)
            dec_emb = self.pos_encoding(dec_emb)
            dec_summary = dec_emb.mean(dim=1, keepdim=True)
            pooled = pooled + 0.1 * dec_summary

        # 扩展到预测长度
        expanded = pooled.expand(batch_size, pred_len, self.d_model)  # (batch_size, pred_len, d_model)
        
        # 输出投影
        output = self.projection(expanded)
        
        return output


# ======================================================================================
# PatchTST - 基于Patch的时序Transformer (ICLR 2023)
# ======================================================================================

# ======================================================================================
# FEDformer - 频域增强Transformer (ICML 2022)
# ======================================================================================

class FrequencyEnhancedAttention(nn.Module):
    """频域增强注意力机制 - 简化版本避免CUDA编译依赖"""
    def __init__(self, d_model, n_heads, modes, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        self.head_dim = d_model // n_heads
        
        # 查询、键、值投影
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # 使用实数权重替代复数权重，避免CUDA编译依赖
        self.freq_weights_real = nn.Parameter(torch.randn(modes, d_model))
        self.freq_weights_imag = nn.Parameter(torch.randn(modes, d_model))
        
        # 频域滤波器
        self.freq_filter = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape
        
        # 线性投影
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # 使用简化的频域增强，避免复数FFT操作
        # 方法1：使用卷积模拟频域滤波
        Q_enhanced = Q.transpose(1, 2)  # [batch, d_model, seq_len]
        K_enhanced = K.transpose(1, 2)
        V_enhanced = V.transpose(1, 2)
        
        Q_enhanced = self.freq_filter(Q_enhanced).transpose(1, 2)  # [batch, seq_len, d_model]
        K_enhanced = self.freq_filter(K_enhanced).transpose(1, 2)
        V_enhanced = self.freq_filter(V_enhanced).transpose(1, 2)
        
        # 方法2：使用可学习的频域权重进行增强
        # 计算简单的频域特征（使用余弦变换近似）
        freq_indices = torch.arange(seq_len, device=Q.device, dtype=torch.float32)
        modes_to_use = min(self.modes, seq_len)
        
        if modes_to_use > 0:
            # 创建频域基函数，形状为 [seq_len, modes_to_use]
            cos_basis = torch.cos(2 * math.pi * freq_indices.unsqueeze(1) * torch.arange(modes_to_use, device=Q.device).unsqueeze(0) / seq_len)
            sin_basis = torch.sin(2 * math.pi * freq_indices.unsqueeze(1) * torch.arange(modes_to_use, device=Q.device).unsqueeze(0) / seq_len)
            
            # 应用频域权重，形状为 [seq_len, d_model]
            freq_real = torch.matmul(cos_basis, self.freq_weights_real[:modes_to_use])  # [seq_len, d_model]
            freq_imag = torch.matmul(sin_basis, self.freq_weights_imag[:modes_to_use])  # [seq_len, d_model]
            freq_enhancement = (freq_real + freq_imag).unsqueeze(0)  # [1, seq_len, d_model]
            
            # 应用频域增强
            Q_enhanced = Q_enhanced + freq_enhancement * 0.1
            K_enhanced = K_enhanced + freq_enhancement * 0.1
            V_enhanced = V_enhanced + freq_enhancement * 0.1
        
        # 多头注意力
        Q_enhanced = Q_enhanced.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K_enhanced = K_enhanced.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V_enhanced = V_enhanced.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q_enhanced, K_enhanced.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_out = torch.matmul(attn_weights, V_enhanced)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_linear(attn_out)


class FEDformerEncoderLayer(nn.Module):
    """FEDformer编码器层 - 包含频域增强注意力"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, modes=32):
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        
        # 频域增强注意力
        self.freq_attention = FrequencyEnhancedAttention(d_model, n_heads, modes, dropout)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # 频域增强注意力
        attn_out = self.freq_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class FEDformer(nn.Module):
    """FEDformer模型实现 - 频域增强Transformer (ICML 2022)"""
    def __init__(self, input_dim=6, d_model=None, n_heads=None, e_layers=None, d_layers=None,
                 d_ff=None, dropout=None, modes=32, seq_len=None, pred_len=None):
        super(FEDformer, self).__init__()
        
        # 动态导入配置以避免循环导入
        from . import HGVConfig
        config = HGVConfig.get_model_config('fedformer')
        self.d_model = d_model or config['d_model']
        self.n_heads = n_heads or config['nhead']
        self.e_layers = e_layers or config['num_encoder_layers']
        self.d_layers = d_layers or config['num_decoder_layers']
        self.d_ff = d_ff or config['dim_feedforward']
        self.dropout = dropout or config['dropout']
        self.modes = modes
        self.seq_len = seq_len or 64
        self.pred_len = pred_len or 256
        self.input_dim = input_dim
        self.output_dim = 3
        
        # 输入嵌入
        self.enc_embedding = nn.Linear(input_dim, self.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)
        
        # 频域增强编码器层
        self.encoder_layers = nn.ModuleList([
            FEDformerEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout, modes)
            for _ in range(self.e_layers)
        ])
        
        # 输出头 - 直接从编码器输出预测
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.pred_len * self.output_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        target_length: int = None,
        decoder_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        前向传播 - 单输入模式
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            target_length: 目标预测长度（可选）
        Returns:
            output: 预测序列 [batch_size, target_length or pred_len, output_dim]
        """
        batch_size = x.size(0)
        
        # 编码器处理
        enc_out = self.enc_embedding(x)  # [batch_size, seq_len, d_model]
        enc_out = self.pos_encoding(enc_out)
        
        # 通过频域增强编码器层
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        
        # 全局平均池化
        enc_out = enc_out.mean(dim=1)  # [batch_size, d_model]

        # decoder context 融合（official-like 协议可选）
        if decoder_context is not None and decoder_context.numel() > 0:
            ctx = decoder_context.mean(dim=1)
            if ctx.size(-1) != self.d_model:
                # 将3维上下文投影到d_model，避免额外模块改动
                pad = torch.zeros(ctx.size(0), self.d_model - ctx.size(-1), device=ctx.device, dtype=ctx.dtype)
                ctx = torch.cat([ctx, pad], dim=-1) if ctx.size(-1) < self.d_model else ctx[:, : self.d_model]
            enc_out = enc_out + 0.1 * ctx
        
        # 输出预测
        output = self.output_head(enc_out)  # [batch_size, pred_len * output_dim]
        output = output.view(batch_size, self.pred_len, self.output_dim)  # [batch_size, pred_len, output_dim]
        
        # 如果指定了target_length，动态调整输出长度
        if target_length is not None and target_length != self.pred_len:
            # 使用插值调整时间维度
            output = output.transpose(1, 2)  # [batch_size, output_dim, pred_len]
            output = torch.nn.functional.interpolate(
                output,
                size=target_length,
                mode='linear',
                align_corners=False
            )
            output = output.transpose(1, 2)  # [batch_size, target_length, output_dim]
        
        return output


# ======================================================================================
# TimesNet - 基于时序2D卷积的Transformer (ICLR 2023)
# ======================================================================================

class TimesBlock(nn.Module):
    """TimesNet的核心TimesBlock"""
    def __init__(self, d_model, d_ff, top_k=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        
        # 2D卷积层
        self.conv2d = nn.Conv2d(1, d_model, kernel_size=(3, 3), padding=(1, 1))
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 简化的周期检测，避免CUDA FFT错误
        # 使用固定的周期列表而不是FFT分析
        candidate_periods = [2, 4, 8, 16, 32]
        period = seq_len // 4  # 使用序列长度的1/4作为默认周期
        
        # 选择最适合的周期
        for p in candidate_periods:
            if p <= seq_len and seq_len % p == 0:
                period = p
                break
        
        period = max(1, min(period, seq_len))
        
        # 重塑为2D
        if seq_len % period == 0:
            x_2d = x.view(batch_size, period, seq_len // period, d_model)
        else:
            # 填充到能被period整除
            pad_len = period - (seq_len % period)
            x_padded = F.pad(x, (0, 0, 0, pad_len))
            x_2d = x_padded.view(batch_size, period, (seq_len + pad_len) // period, d_model)
        
        # 2D卷积处理
        x_2d_processed = []
        for i in range(d_model):
            x_channel = x_2d[:, :, :, i].unsqueeze(1)  # [batch, 1, period, time_steps]
            conv_out = self.conv2d(x_channel)  # [batch, d_model, period, time_steps]
            x_2d_processed.append(conv_out.mean(dim=1))  # 平均池化到原始维度
        
        x_2d_out = torch.stack(x_2d_processed, dim=-1)  # [batch, period, time_steps, d_model]
        
        # 重塑回1D
        x_out = x_2d_out.view(batch_size, -1, d_model)[:, :seq_len, :]
        
        # 残差连接和层归一化
        x = self.norm1(x + x_out)
        
        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class TimesNet(nn.Module):
    """TimesNet模型实现"""
    def __init__(self, input_dim=6, d_model=None, n_heads=None, e_layers=None, d_layers=None,
                 d_ff=None, dropout=None, top_k=5, seq_len=None, pred_len=None):
        super(TimesNet, self).__init__()
        
        # 动态导入配置以避免循环导入
        from . import HGVConfig
        config = HGVConfig.get_model_config('timesnet')
        self.d_model = d_model or config['d_model']
        self.n_heads = n_heads or config['nhead']
        self.e_layers = e_layers or config['num_encoder_layers']
        self.d_ff = d_ff or config['dim_feedforward']
        self.dropout = dropout or config['dropout']
        self.top_k = top_k
        self.seq_len = seq_len or 64
        self.pred_len = pred_len or 256
        
        # 输入嵌入
        self.enc_embedding = nn.Linear(input_dim, self.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)
        
        # TimesBlock编码器层
        self.encoder_layers = nn.ModuleList([
            TimesBlock(self.d_model, self.d_ff, top_k, self.dropout)
            for _ in range(self.e_layers)
        ])
        
        # 输出投影 - 先投影到预测长度
        self.temporal_projection = nn.Linear(self.seq_len, self.pred_len)
        # 然后投影到输出维度
        self.feature_projection = nn.Linear(self.d_model, 3)
        
    def forward(self, x, target_length: int = None, decoder_context: torch.Tensor | None = None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            target_length: 目标预测长度（可选）
        Returns:
            Output tensor of shape (batch_size, target_length or pred_len, output_dim)
        """
        # 编码器
        enc_out = self.enc_embedding(x)
        enc_out = self.pos_encoding(enc_out)
        
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        
        # 时间维度投影：从seq_len投影到pred_len
        # enc_out: (batch_size, seq_len, d_model)
        enc_out = enc_out.transpose(1, 2)  # (batch_size, d_model, seq_len)
        enc_out = self.temporal_projection(enc_out)  # (batch_size, d_model, pred_len)
        enc_out = enc_out.transpose(1, 2)  # (batch_size, pred_len, d_model)
        
        # 特征维度投影：从d_model投影到4
        output = self.feature_projection(enc_out)
        
        # 如果指定了target_length，动态调整输出长度
        if target_length is not None and target_length != self.pred_len:
            output = output.transpose(1, 2)
            output = torch.nn.functional.interpolate(
                output,
                size=target_length,
                mode='linear',
                align_corners=False
            )
            output = output.transpose(1, 2)
        
        return output


# ======================================================================================
# iTransformer - 倒置Transformer (NeurIPS 2024)
# ======================================================================================


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    
    Reference: https://arxiv.org/abs/2310.06625
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 d_model: int = None,
                 nhead: int = None,
                 num_encoder_layers: int = None,
                 num_decoder_layers: int = None,
                 dim_feedforward: int = None,
                 dropout: float = None,
                 seq_len: int = None,
                 pred_len: int = None):
        super().__init__()
        
        # 动态导入配置以避免循环导入
        from . import HGVConfig
        config = HGVConfig.get_model_config('itransformer')
        
        self.input_dim = input_dim or config['input_dim']
        self.d_model = d_model or config['d_model']
        # 从训练配置中获取序列长度参数
        train_config = HGVConfig.get_train_config()
        self.seq_len = seq_len or train_config['seq_len']
        self.pred_len = pred_len or train_config['pred_len']
        
        # 使用配置中的参数
        nhead = nhead or config['nhead']
        num_encoder_layers = num_encoder_layers or config['num_encoder_layers']
        dim_feedforward = dim_feedforward or config['dim_feedforward']
        dropout = dropout or config['dropout']
        
        # Inverted embedding: treat each variate as a token
        self.variate_embedding = nn.Linear(self.seq_len, self.d_model)
        self.positional_encoding = nn.Parameter(torch.randn(self.input_dim, self.d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Output projection - 输出3维而不是input_dim维
        self.output_projection = nn.Linear(self.d_model, self.pred_len)
        self.final_projection = nn.Linear(self.input_dim, 3)
        
    def forward(
        self,
        x: torch.Tensor,
        target_length: int = None,
        decoder_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            target_length: 目标预测长度（可选）
        Returns:
            Output tensor of shape (batch_size, target_length or pred_len, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Transpose to treat variates as tokens: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Variate embedding: (batch_size, input_dim, d_model)
        x = self.variate_embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding.unsqueeze(0)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Output projection: (batch_size, input_dim, pred_len)
        x = self.output_projection(x)
        
        # Transpose back: (batch_size, pred_len, input_dim)
        x = x.transpose(1, 2)
        
        # 如果指定了target_length，动态调整输出长度
        if target_length is not None and target_length != self.pred_len:
            x = self.final_projection(x)
            x = x.transpose(1, 2)
            x = torch.nn.functional.interpolate(
                x,
                size=target_length,
                mode='linear',
                align_corners=False
            )
            x = x.transpose(1, 2)
            return x
        
        # 否则检查是否需要硬编码的长度调整（旧逻辑保留兼容性）
        target_len = self.pred_len
        if x.shape[1] != target_len:
            if x.shape[1] < target_len:
                # 如果输出长度不足，进行填充
                pad_len = target_len - x.shape[1]
                x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
            else:
                # 如果输出长度过长，进行截取
                x = x[:, :target_len, :]
        
        x = self.final_projection(x)
        
        return x


class PatchTST(nn.Module):
    """
    PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
    
    Reference: https://arxiv.org/abs/2211.14730
    """
    
    def __init__(self,
                 input_dim: int = None,
                 d_model: int = None,
                 nhead: int = None,
                 num_layers: int = None,
                 dim_feedforward: int = None,
                 dropout: float = None,
                 seq_len: int = None,
                 pred_len: int = None,
                 patch_len: int = None,
                 stride: int = None):
        super().__init__()
        
        # 动态导入配置以避免循环导入
        from . import HGVConfig
        config = HGVConfig.get_model_config('patchtst')
        
        self.input_dim = input_dim or config['input_dim']
        self.d_model = d_model or config['d_model']
        # 从训练配置中获取序列长度参数
        train_config = HGVConfig.get_train_config()
        self.seq_len = seq_len or train_config['seq_len']
        self.pred_len = pred_len or train_config['pred_len']
        self.patch_len = patch_len or config.get('patch_len', 16)
        self.stride = stride or config.get('stride', 8)
        
        self.output_dim = 3
        
        # 使用配置中的参数
        nhead = nhead or config['nhead']
        num_layers = num_layers or config['num_encoder_layers']
        dim_feedforward = dim_feedforward or config['dim_feedforward']
        dropout = dropout or config['dropout']
        
        # Calculate number of patches
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1
        
        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(self.num_patches, self.d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Channel-independent PatchTST head shared across variates.
        self.output_head = nn.Linear(self.num_patches * self.d_model, self.pred_len)
        
    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create patches from input time series
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Patches tensor of shape (batch_size, input_dim, num_patches, patch_len)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Transpose to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, :, i:i + self.patch_len]
            patches.append(patch)
        
        # Stack patches: (batch_size, input_dim, num_patches, patch_len)
        patches = torch.stack(patches, dim=2)
        
        return patches
    
    def forward(
        self,
        x: torch.Tensor,
        target_length: int = None,
        decoder_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            target_length: 目标预测长度（可选）。如果提供，输出将调整到此长度
        Returns:
            Output tensor of shape (batch_size, target_length or pred_len-1, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Create patches: (batch_size, input_dim, num_patches, patch_len)
        patches = self.create_patches(x)
        
        # 动态计算实际的patch数量
        actual_num_patches = patches.size(2)
        
        outputs = []
        for i in range(input_dim):
            # Get patches for current variate: (batch_size, num_patches, patch_len)
            variate_patches = patches[:, i, :, :]
            
            # Patch embedding: (batch_size, num_patches, d_model)
            embedded = self.patch_embedding(variate_patches)
            
            # Add positional encoding - 动态截取或扩展以匹配实际patch数量
            if actual_num_patches <= self.num_patches:
                pos_encoding = self.positional_encoding[:actual_num_patches, :]
            else:
                # 如果实际patch数多于预设，使用插值扩展
                pos_encoding = torch.nn.functional.interpolate(
                    self.positional_encoding.unsqueeze(0).transpose(1, 2),
                    size=actual_num_patches,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2).squeeze(0)
            
            embedded = embedded + pos_encoding.unsqueeze(0)
            
            # Transformer encoding
            encoded = self.transformer_encoder(embedded)
            
            # Flatten and project: (batch_size, (pred_len-1) * output_dim)
            flattened = encoded.reshape(batch_size, -1)
            
            # 动态处理维度不匹配：如果实际维度与预期不同，进行自适应投影
            if flattened.size(1) != self.num_patches * self.d_model:
                # 使用自适应平均池化调整到正确的维度
                flattened = flattened.view(batch_size, actual_num_patches, self.d_model)
                # 池化到预期的patch数量
                if actual_num_patches < self.num_patches:
                    # 如果实际patch少，进行上采样（复制）
                    flattened = torch.nn.functional.interpolate(
                        flattened.transpose(1, 2),
                        size=self.num_patches,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
                else:
                    # 如果实际patch多，进行下采样
                    flattened = torch.nn.functional.adaptive_avg_pool1d(
                        flattened.transpose(1, 2),
                        self.num_patches
                    ).transpose(1, 2)
                flattened = flattened.reshape(batch_size, -1)
            
            output = self.output_head(flattened)
            
            outputs.append(output)
        
        # The first three variates are the position channels in target scaling.
        final_output = torch.stack(outputs[:self.output_dim], dim=-1)
        
        # 如果指定了target_length，动态调整输出长度
        if target_length is not None and target_length != self.pred_len:
            # 使用插值调整时间维度到目标长度
            # final_output shape: (batch_size, pred_len, output_dim)
            final_output = final_output.transpose(1, 2)
            # 插值到target_length
            final_output = torch.nn.functional.interpolate(
                final_output,
                size=target_length,
                mode='linear',
                align_corners=False
            )
            # 转换回 (batch_size, target_length, output_dim)
            final_output = final_output.transpose(1, 2)
        
        return final_output


def create_sota_model(model_type: str, input_dim=6, device=torch.device('cpu'), **kwargs) -> nn.Module:
    """
    Factory function to create SOTA models
    
    Args:
        model_type: Type of model to create
        input_dim: Input dimension
        device: Device to place model on
        **kwargs: Model-specific parameters
    
    Returns:
        Initialized model instance
    """
    # 获取统一配置
    from . import HGVConfig
    base_config = HGVConfig.get_model_config(model_type)
    base_config['input_dim'] = input_dim
    
    # 获取训练配置中的序列长度参数
    train_config = HGVConfig.get_train_config()
    base_config['seq_len'] = train_config.get('seq_len', 64)
    base_config['pred_len'] = train_config.get('pred_len', 256)
    
    # 允许通过kwargs覆盖配置
    base_config.update(kwargs)
    
    # 参数名称映射 - 将HGVConfig的参数名映射到各SOTA模型期望的参数名
    if model_type.lower() in ['informer', 'autoformer', 'fedformer', 'timesnet']:
        # 这些模型使用 n_heads, e_layers, d_layers, d_ff
        model_config = {
            'input_dim': base_config.get('input_dim', input_dim),
            'd_model': base_config.get('d_model'),
            'n_heads': base_config.get('nhead'),  # nhead -> n_heads
            'e_layers': base_config.get('num_encoder_layers'),  # num_encoder_layers -> e_layers
            'd_layers': base_config.get('num_decoder_layers'),  # num_decoder_layers -> d_layers
            'd_ff': base_config.get('dim_feedforward'),  # dim_feedforward -> d_ff
            'dropout': base_config.get('dropout'),
        }
        # 添加模型特定参数
        if model_type.lower() == 'informer':
            model_config['factor'] = base_config.get('factor', 5)
        elif model_type.lower() == 'autoformer':
            model_config['moving_avg'] = base_config.get('moving_avg', 25)
            model_config['factor'] = base_config.get('factor', 3)
        elif model_type.lower() == 'fedformer':
            model_config['modes'] = base_config.get('modes', 32)
        elif model_type.lower() == 'timesnet':
            model_config['top_k'] = base_config.get('top_k', 5)
            
    elif model_type.lower() in ['patchtst', 'itransformer']:
        # 这些模型使用标准参数名
        model_config = {
            'input_dim': base_config.get('input_dim', input_dim),
            'd_model': base_config.get('d_model'),
            'nhead': base_config.get('nhead'),
            'dropout': base_config.get('dropout'),
        }
        # 添加模型特定参数
        if model_type.lower() == 'patchtst':
            model_config.update({
                'num_layers': base_config.get('num_encoder_layers'),
                'dim_feedforward': base_config.get('dim_feedforward'),
                'seq_len': base_config.get('seq_len'),  # 使用配置中的实际值
                'pred_len': base_config.get('pred_len'),  # 使用配置中的实际值
                'patch_len': base_config.get('patch_len', 16),
                'stride': base_config.get('stride', 8),
            })
        elif model_type.lower() == 'itransformer':
            model_config.update({
                'num_encoder_layers': base_config.get('num_encoder_layers'),
                'num_decoder_layers': base_config.get('num_decoder_layers'),
                'dim_feedforward': base_config.get('dim_feedforward'),
                'seq_len': base_config.get('seq_len'),  # 使用配置中的实际值
                'pred_len': base_config.get('pred_len'),  # 使用配置中的实际值
            })
    else:
        # 默认使用原始配置
        model_config = base_config
    
    model_registry = {
        'informer': Informer,
        'autoformer': Autoformer,
        'patchtst': PatchTST,
        'fedformer': FEDformer,
        'timesnet': TimesNet,
        'itransformer': iTransformer,
    }
    
    if model_type.lower() not in model_registry:
        raise ValueError(f"Unknown SOTA model type: {model_type}. Available: {list(model_registry.keys())}")
    
    model_class = model_registry[model_type.lower()]
    model = model_class(**model_config)
    
    return model.to(device)


# Export all models
__all__ = [
    'Informer',
    'Autoformer',
    'PatchTST', 
    'FEDformer',
    'TimesNet',
    'iTransformer',
    'create_sota_model'
]
