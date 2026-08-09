#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLGAFormer model and dimensionless physics-aware training objective.

The validation-selected model uses the gated rotating-Earth prior fusion and
multi-head trajectory decoder (C). Physics-aware attention (A) and the optional
kinematic corrector (B) are retained only as ablation candidates because the
selection experiments did not support including them in the primary model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import hashlib


EARTH_RADIUS_M = 6_378_000.0
EARTH_MU_M3_S2 = 3.986004418e14


def spherical_to_cartesian_torch(position):
    """Convert ``[r, longitude, latitude]`` to Earth-centered Cartesian SI coordinates."""
    radius = position[..., 0]
    longitude = position[..., 1]
    latitude = position[..., 2]
    cos_latitude = torch.cos(latitude)
    return torch.stack(
        (
            radius * cos_latitude * torch.cos(longitude),
            radius * cos_latitude * torch.sin(longitude),
            radius * torch.sin(latitude),
        ),
        dim=-1,
    )


def cartesian_to_spherical_torch(position):
    """Convert Earth-centered Cartesian SI coordinates to ``[r, longitude, latitude]``."""
    radius = torch.linalg.norm(position, dim=-1).clamp_min(1.0)
    longitude = torch.atan2(position[..., 1], position[..., 0])
    latitude = torch.asin((position[..., 2] / radius).clamp(-1.0, 1.0))
    return torch.stack((radius, longitude, latitude), dim=-1)


class ECEFTrajectoryLoss(nn.Module):
    """Dimensionless spatial loss aligned with the reported ECEF trajectory error."""

    def __init__(
        self,
        scaler_mean,
        scaler_scale,
        distance_scale_m=100_000.0,
        scaled_mse_weight=0.1,
    ):
        super().__init__()
        if distance_scale_m <= 0.0:
            raise ValueError("distance_scale_m must be positive.")
        if scaled_mse_weight < 0.0:
            raise ValueError("scaled_mse_weight must be non-negative.")
        mean = torch.as_tensor(scaler_mean, dtype=torch.float32)
        scale = torch.as_tensor(scaler_scale, dtype=torch.float32)
        if mean.numel() < 3 or scale.numel() < 3:
            raise ValueError("ECEFTrajectoryLoss requires three-channel output scaler metadata.")
        self.register_buffer("scaler_mean", mean[:3])
        self.register_buffer("scaler_scale", scale[:3])
        self.distance_scale_m = float(distance_scale_m)
        self.scaled_mse_weight = float(scaled_mse_weight)
        self.last_components = None

    def forward(self, pred, target):
        if pred.shape != target.shape or pred.size(-1) < 3:
            raise ValueError(
                f"pred and target must share [..., 3] shapes; got {pred.shape} and {target.shape}."
            )
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_scaled = pred.float()[..., :3]
            target_scaled = target.float()[..., :3]
            mean = self.scaler_mean.to(device=pred.device)
            scale = self.scaler_scale.to(device=pred.device)
            pred_physical = pred_scaled * scale + mean
            target_physical = target_scaled * scale + mean
            pred_ecef = spherical_to_cartesian_torch(pred_physical)
            target_ecef = spherical_to_cartesian_torch(target_physical)
            ecef_mse = F.mse_loss(
                pred_ecef / self.distance_scale_m,
                target_ecef / self.distance_scale_m,
            )
            scaled_mse = F.mse_loss(pred_scaled, target_scaled)
            total = ecef_mse + self.scaled_mse_weight * scaled_mse
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite ECEF trajectory loss.")
        self.last_components = {
            "ecef_mse": float(ecef_mse.detach().cpu()),
            "scaled_mse": float(scaled_mse.detach().cpu()),
        }
        return total

# 延迟导入HGVConfig以避免循环导入
def get_hgv_config():
    """延迟导入HGVConfig以避免循环导入"""
    try:
        from . import HGVConfig
        return HGVConfig
    except ImportError:
        # 如果相对导入失败，尝试绝对导入
        from models import HGVConfig
        return HGVConfig


def _stable_module_seed(base_seed, module_name):
    digest = hashlib.sha256(f"{int(base_seed)}:{module_name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


class HGVPhysicsLoss(nn.Module):
    """Dimensionless trajectory loss for the spherical-Earth 3-DOF state model."""

    def __init__(
        self,
        alpha=None,
        acceleration_weight=None,
        beta=None,
        scaler_mean=None,
        scaler_std=None,
    ):
        super().__init__()
        # 从统一配置获取参数
        HGVConfig = get_hgv_config()
        physics_config = HGVConfig.get_physics_config()
        physical_constraints = HGVConfig.get_physical_constraints()
        
        self.alpha = float(alpha if alpha is not None else physics_config['alpha'])
        if acceleration_weight is not None and beta is not None:
            raise ValueError("Use acceleration_weight or the legacy beta alias, not both.")
        if acceleration_weight is None:
            acceleration_weight = beta
        self.acceleration_weight = float(
            acceleration_weight
            if acceleration_weight is not None
            else physics_config['acceleration_weight']
        )
        if self.alpha < 0.0 or self.acceleration_weight < 0.0:
            raise ValueError("Physics-loss weights must be non-negative.")
        self.mse_loss = nn.MSELoss()
        
        # 存储标准化参数用于反归一化
        self.enable_denorm = False
        if scaler_mean is not None and scaler_std is not None:
            self.register_buffer('scaler_mean', torch.as_tensor(scaler_mean, dtype=torch.float32))
            self.register_buffer('scaler_std', torch.as_tensor(scaler_std, dtype=torch.float32))
            self.enable_denorm = True
            
        # 从统一配置读取物理约束参数（确保与数据生成器一致）
        self.min_height = physical_constraints['min_height']           # 最小高度约束 (m)
        self.max_height = physical_constraints['max_height']           # 最大高度约束 (m)
        self.max_gamma = physical_constraints['max_gamma']             # 最大航迹角约束 (rad)
        self.max_latitude = physical_constraints['max_latitude']       # 最大纬度约束 (rad)
        self.max_azimuth_rate = physical_constraints['max_azimuth_rate']  # 最大方位角变化率约束 (rad/s)
        
        # HGV物理常数（从统一配置读取）
        self.g = physical_constraints['g']                             # 重力加速度 (m/s²)
        self.earth_radius = physical_constraints['earth_radius']       # 地球半径 (m)
        self.last_components = None
        
    def forward(self, pred, target, input_data=None, dt=1.0, include_mse=True):
        """Compute a dimensionless, physically interpretable trajectory loss.

        Position constraints are normalized by their admissible ranges. When
        output-scaler metadata are available, first- and second-order residuals
        are evaluated in Earth-centered Cartesian SI coordinates and normalized
        by target velocity/acceleration scales. This avoids adding meters,
        radians, and normalized MSE directly.
        """
        del input_data
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have identical shapes; got {pred.shape} and {target.shape}.")
        if pred.ndim != 3 or pred.size(-1) < 3:
            raise ValueError("HGVPhysicsLoss expects [batch, horizon, >=3] trajectories.")

        mse_loss = self.mse_loss(pred, target) if include_mse else pred.new_zeros(())
        if pred.size(1) <= 1:
            return mse_loss

        if self.enable_denorm:
            mean = self.scaler_mean.to(device=pred.device, dtype=pred.dtype)
            scale = self.scaler_std.to(device=pred.device, dtype=pred.dtype)
            pred_physical = pred * scale + mean
            target_physical = target * scale + mean

            height_range = max(float(self.max_height - self.min_height), 1.0)
            pred_height = pred_physical[..., 0] - float(self.earth_radius)
            lower_violation = F.relu((float(self.min_height) - pred_height) / height_range)
            upper_violation = F.relu((pred_height - float(self.max_height)) / height_range)
            height_loss = (lower_violation.square() + upper_violation.square()).mean()

            latitude_scale = max(float(self.max_latitude), 1e-6)
            latitude_violation = F.relu(
                (pred_physical[..., 2].abs() - float(self.max_latitude)) / latitude_scale
            )
            latitude_loss = latitude_violation.square().mean()

            pred_xyz = spherical_to_cartesian_torch(pred_physical[..., :3])
            target_xyz = spherical_to_cartesian_torch(target_physical[..., :3])
            dt_value = max(float(dt), 1e-6)
            pred_velocity = torch.diff(pred_xyz, dim=1) / dt_value
            target_velocity = torch.diff(target_xyz, dim=1) / dt_value
            velocity_scale = target_velocity.square().mean().sqrt().detach().clamp_min(100.0)
            velocity_residual = (pred_velocity - target_velocity) / velocity_scale
            first_order_loss = torch.log1p(velocity_residual.square()).mean()

            if pred.size(1) > 2:
                pred_acceleration = torch.diff(pred_velocity, dim=1) / dt_value
                target_acceleration = torch.diff(target_velocity, dim=1) / dt_value
                acceleration_scale = target_acceleration.square().mean().sqrt().detach().clamp_min(float(self.g))
                acceleration_residual = (pred_acceleration - target_acceleration) / acceleration_scale
                second_order_loss = torch.log1p(acceleration_residual.square()).mean()
            else:
                second_order_loss = pred.new_zeros(())
        else:
            height_loss = pred.new_zeros(())
            latitude_loss = pred.new_zeros(())
            first_order_loss = F.mse_loss(torch.diff(pred, dim=1), torch.diff(target, dim=1))
            second_order_loss = (
                F.mse_loss(torch.diff(pred, n=2, dim=1), torch.diff(target, n=2, dim=1))
                if pred.size(1) > 2
                else pred.new_zeros(())
            )

        physics_loss = (
            height_loss
            + latitude_loss
            + first_order_loss
            + self.acceleration_weight * second_order_loss
        )
        weighted_physics = self.alpha * physics_loss
        total_loss = mse_loss + weighted_physics
        self.last_components = {
            "mse": mse_loss.detach(),
            "height": height_loss.detach(),
            "latitude": latitude_loss.detach(),
            "velocity_cauchy": first_order_loss.detach(),
            "acceleration_cauchy": second_order_loss.detach(),
            "acceleration_weight": pred.new_tensor(self.acceleration_weight),
            "physics_residual": physics_loss.detach(),
            "weighted_physics": weighted_physics.detach(),
            "weighted_total": total_loss.detach(),
        }
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite HGV physics loss.")
        return total_loss

# ==============================================================================
# 创新点 A：物理感知注意力 (Physics-aware Attention)
# ==============================================================================

class PhysicsAwareAttention(nn.Module):
    """
    HGV 物理感知多头注意力机制（创新点 A）

    在标准 self-attention 基础上注入三类物理先验偏置，使注意力权重
    不仅由数据驱动，还受 HGV 飞行力学先验约束：

    1. 时间衰减偏置 (Temporal Decay Bias)
       B_td(i,j) = -α_h · |i − j|
       每个注意力头学习独立衰减速率 α_h，反映 HGV 轨迹的时间连续性：
       近邻时刻的状态转移更具预测参考价值。

    2. 飞行阶段一致性偏置 (Phase Coherence Bias)
       利用高度、速度、航迹角和比机械能构造显式阶段特征，
       使动力学状态相近的时间步相互关注更强。

    3. 几何邻近性偏置 (Geometric Proximity Bias)
       将球坐标位置 [r, λ, φ] 转到地心地固笛卡尔坐标，
       物理空间中相近的状态获得更强注意力。

    最终注意力:
      Attention(Q,K,V) = softmax(QK^T/√d + B_td + B_phase + B_geo) · V
    """

    def __init__(self, d_model, nhead, dropout=0.1, prior_mask=(True, True, True)):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        if len(prior_mask) != 3:
            raise ValueError("prior_mask must contain temporal, phase, and geometry flags.")
        self.prior_mask = tuple(bool(value) for value in prior_mask)

        # Each head learns only the bandwidth and strength of explicit priors.
        self.temporal_decay_raw = nn.Parameter(torch.zeros(nhead))
        self.phase_bandwidth_raw = nn.Parameter(torch.zeros(nhead))
        self.geo_bandwidth_raw = nn.Parameter(torch.zeros(nhead))
        self.bias_scale = nn.Parameter(torch.full((3,), -3.0))
        self.register_buffer("earth_radius_m", torch.tensor(EARTH_RADIUS_M, dtype=torch.float32))
        self.register_buffer("earth_mu_m3_s2", torch.tensor(EARTH_MU_M3_S2, dtype=torch.float32))
        self.last_bias_diagnostics = None

    def forward(self, x, raw_input=None):
        """
        Args:
            x: 嵌入后的特征 [batch, seq_len, d_model]
            raw_input: 原始物理输入 [batch, seq_len, 6] — [r, λ, φ, V, γ, ψ]
        Returns:
            [batch, seq_len, d_model]
        """
        B, L, _ = x.shape
        H, D = self.nhead, self.head_dim
        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, D).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if raw_input is not None:
            if raw_input.size(-1) < 6:
                raise ValueError("Physics-aware attention requires [r, lon, lat, V, gamma, psi].")

            # Earth radius and gravitational-parameter magnitudes overflow
            # FP16. Construct only the physical bias in FP32, then cast the
            # bounded, dimensionless bias back to the AMP score dtype.
            with torch.autocast(device_type=x.device.type, enabled=False):
                raw_fp32 = raw_input.float()
                active_mask = torch.as_tensor(self.prior_mask, device=x.device, dtype=torch.float32)
                scales = torch.sigmoid(self.bias_scale.float()) * active_mask
                decay = F.softplus(self.temporal_decay_raw.float())
                phase_bandwidth = F.softplus(self.phase_bandwidth_raw.float()).clamp_min(1e-4)
                geo_bandwidth = F.softplus(self.geo_bandwidth_raw.float()).clamp_min(1e-4)
                physical_bias = torch.zeros(B, H, L, L, device=x.device, dtype=torch.float32)

                pos = torch.arange(L, device=x.device, dtype=torch.float32)
                dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
                dist = dist / float(max(L - 1, 1))
                if self.prior_mask[0]:
                    temporal_bias = -dist.unsqueeze(0) * decay.view(H, 1, 1)
                    physical_bias = physical_bias + scales[0] * temporal_bias.unsqueeze(0)

                earth_radius = self.earth_radius_m.float()
                earth_mu = self.earth_mu_m3_s2.float()
                radius = raw_fp32[..., 0].clamp_min(1.0)
                speed = raw_fp32[..., 3]
                flight_path = raw_fp32[..., 4]
                altitude_ratio = (radius - earth_radius) / 100_000.0
                speed_ratio = speed / 8_000.0
                energy_ratio = (0.5 * speed.square() - earth_mu / radius) / (earth_mu / earth_radius)
                phase_features = torch.stack(
                    (altitude_ratio, speed_ratio, torch.sin(flight_path), energy_ratio), dim=-1
                )
                phase_distance = torch.cdist(phase_features, phase_features, p=2).square()
                if self.prior_mask[1]:
                    phase_bias = -torch.clamp(
                        phase_distance.unsqueeze(1) / phase_bandwidth.view(1, H, 1, 1).square(),
                        max=4.0,
                    )
                    physical_bias = physical_bias + scales[1] * phase_bias

                ecef = spherical_to_cartesian_torch(raw_fp32[..., :3]) / earth_radius
                geo_distance = torch.cdist(ecef, ecef, p=2)
                if self.prior_mask[2]:
                    # A 500-km chord is the reference spatial scale. Normalizing
                    # before applying the learned bandwidth keeps this logit prior
                    # comparable across input lengths and geographic locations.
                    normalized_geo_distance = geo_distance / (500_000.0 / float(EARTH_RADIUS_M))
                    geo_bias = -torch.clamp(
                        normalized_geo_distance.unsqueeze(1) / geo_bandwidth.view(1, H, 1, 1),
                        max=4.0,
                    )
                    physical_bias = physical_bias + scales[2] * geo_bias

                if not torch.isfinite(physical_bias).all():
                    raise FloatingPointError("Non-finite explicit physics attention bias.")
                self.last_bias_diagnostics = {
                    "temporal_gate": scales[0].detach(),
                    "phase_gate": scales[1].detach(),
                    "geometry_gate": scales[2].detach(),
                    "temporal_decay": decay.detach().mean(),
                    "phase_bandwidth": phase_bandwidth.detach().mean(),
                    "geometry_bandwidth": geo_bandwidth.detach().mean(),
                }
            scores = scores + physical_bias.to(dtype=scores.dtype)

        attn_weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class PhysicsAwareEncoderLayer(nn.Module):
    """物理感知编码器层 (Pre-LN)：PhysicsAwareAttention + FFN"""

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, prior_mask=(True, True, True)):
        super().__init__()
        self.self_attn = PhysicsAwareAttention(d_model, nhead, dropout, prior_mask=prior_mask)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention_residual_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, raw_input=None):
        # Pre-LN (norm_first=True)，与 baseline StandardTransformer 保持一致
        x2 = self.norm1(x)
        x = x + self.attention_residual_dropout(self.self_attn(x2, raw_input=raw_input))
        x2 = self.norm2(x)
        x = x + self.ffn(x2)
        return x


class PhysicsAwareEncoder(nn.Module):
    """物理感知编码器：逐层传递原始物理输入以计算物理偏置"""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, raw_input=None):
        for layer in self.layers:
            x = layer(x, raw_input=raw_input)
        return x


class PositionalEncoding(nn.Module):
    """物理感知位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # 标准正弦位置编码
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                           (-math.log(10000.0) / d_model))
        
        sin_vals = torch.sin(position * div_term)
        cos_vals = torch.cos(position * div_term)
        
        sin_vals = torch.where(torch.isfinite(sin_vals), sin_vals, torch.zeros_like(sin_vals))
        cos_vals = torch.where(torch.isfinite(cos_vals), cos_vals, torch.zeros_like(cos_vals))
        
        pe[:, 0::2] = sin_vals
        pe[:, 1::2] = cos_vals
        
        pe = torch.where(torch.isfinite(pe), pe, torch.zeros_like(pe))
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x, phase_info=None):
        seq_len = x.size(1)
        
        pos_encoding = self.pe[:seq_len, :].transpose(0, 1)
        
        if torch.isnan(pos_encoding).any() or torch.isinf(pos_encoding).any():
            pos_encoding = torch.zeros_like(pos_encoding)
        
        x = x + pos_encoding
        return self.dropout(x)


class PLGAFormerTransformer(nn.Module):
    """
    PLGAFormer: Physics-aware Long-range Glide vehicle Attention Former

    核心创新点：
    A. 物理感知注意力编码器 (Physics-aware Attention Encoder)
       在 self-attention 中注入时间衰减 / 飞行阶段一致性 / 几何邻近性三类物理偏置。
    B. 门控物理校正器 (Gated Adaptive Physics Corrector)
       解码器输出上施加可学习门控残差物理校正。
    C. 多头轨迹解码器 (Multi-Head Trajectory Decoder)
       位置专用头 + 门控融合的多头输出投影。

    消融开关：use_sparse_attention (A)、use_physics_corrector (B)、use_multi_head_output (C)。
    """
    def __init__(self, input_dim=None, d_model=None, nhead=None, num_encoder_layers=None, 
                 num_decoder_layers=None, dim_feedforward=None, dropout=None,
                 # 消融实验参数 - 三个创新点的开关
                 use_sparse_attention=False, use_physics_corrector=False,
                 use_multi_head_output=True, use_adaptive_fusion=True,
                 use_prior_fusion=True, use_channel_residual=False,
                 prior_type='rotating_3dof', prior_blend_mode='adaptive',
                 # 输出维度参数
                 output_dim=3,
                 input_scaler_mean=None, input_scaler_scale=None,
                 output_scaler_mean=None, output_scaler_scale=None,
                 sampling_interval_s=1.0, require_physical_scaler=False,
                 attention_prior_mask=(True, True, True),
                 # 兼容性参数
                 physics_aware_level='full', enable_long_range=True, 
                 glide_specific_features=True):
        super().__init__()
        self._init_seed = int(torch.initial_seed())
        
        # 从统一配置获取参数
        HGVConfig = get_hgv_config()
        model_config = HGVConfig.get_model_config('plgaformer')
        
        # 使用传入参数或配置默认值
        self.input_dim = input_dim if input_dim is not None else model_config['input_dim']
        self.d_model = d_model if d_model is not None else model_config['d_model']
        self.nhead = nhead if nhead is not None else model_config['nhead']
        self.num_encoder_layers = num_encoder_layers if num_encoder_layers is not None else model_config['num_encoder_layers']
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else model_config['num_decoder_layers']
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else model_config['dim_feedforward']
        self.dropout = dropout if dropout is not None else model_config['dropout']
        self.output_dim = output_dim
        
        # 三个创新点的开关
        self.use_sparse_attention = use_sparse_attention        # 创新点A
        self.use_physics_corrector = use_physics_corrector      # 创新点B  
        self.use_multi_head_output = use_multi_head_output      # 创新点C
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_prior_fusion = bool(use_prior_fusion)
        self.use_channel_residual = bool(use_channel_residual)
        self.prior_type = str(prior_type).strip().lower()
        self.prior_blend_mode = str(prior_blend_mode).strip().lower()
        if self.prior_type not in {'rotating_3dof', 'spherical_kinematic'}:
            raise ValueError(
                "prior_type must be 'rotating_3dof' or 'spherical_kinematic'."
            )
        if self.prior_blend_mode not in {'adaptive', 'schedule_only'}:
            raise ValueError(
                "prior_blend_mode must be 'adaptive' or 'schedule_only'."
            )
        self.kinematic_velocity_clip = 1.0
        self.physics_prior_confidence_sharpness = 2.0
        self.physics_prior_disagreement_sharpness = 0.5
        self.physics_prior_time_constant_s = 450.0
        self.physics_prior_decay_power = 2.0
        self.physics_fusion_logit_max = 0.5
        self.sampling_interval_s = float(sampling_interval_s)
        self.require_physical_scaler = bool(require_physical_scaler)
        if len(attention_prior_mask) != 3:
            raise ValueError("attention_prior_mask must contain temporal, phase, and geometry flags.")
        self.attention_prior_mask = tuple(bool(value) for value in attention_prior_mask)
        self.register_buffer(
            "input_scaler_mean",
            torch.as_tensor(input_scaler_mean, dtype=torch.float32)
            if input_scaler_mean is not None
            else torch.empty(0, dtype=torch.float32),
        )
        self.register_buffer(
            "input_scaler_scale",
            torch.as_tensor(input_scaler_scale, dtype=torch.float32)
            if input_scaler_scale is not None
            else torch.empty(0, dtype=torch.float32),
        )
        self.register_buffer(
            "output_scaler_mean",
            torch.as_tensor(output_scaler_mean, dtype=torch.float32)
            if output_scaler_mean is not None
            else torch.empty(0, dtype=torch.float32),
        )
        self.register_buffer(
            "output_scaler_scale",
            torch.as_tensor(output_scaler_scale, dtype=torch.float32)
            if output_scaler_scale is not None
            else torch.empty(0, dtype=torch.float32),
        )
        
        # ==================== 输入处理 ====================
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        self.output_embedding = nn.Linear(self.output_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        
        # ==================== 编码器 ====================
        # 标准 Transformer 编码器始终存在。创新点 A 只作为有界物理先验残差
        # 叠加在同一主干上，保证消融时可以严格退化到 baseline。
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.nhead, self.dim_feedforward, self.dropout,
            batch_first=True, activation='gelu', norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)

        if self.use_sparse_attention:
            # 创新点 A：物理感知注意力编码器
            physics_layers = [
                PhysicsAwareEncoderLayer(
                    self.d_model,
                    self.nhead,
                    self.dim_feedforward,
                    self.dropout,
                    prior_mask=self.attention_prior_mask,
                )
                for _ in range(self.num_encoder_layers)
            ]
            self.physics_encoder = PhysicsAwareEncoder(physics_layers)
        
        # ==================== 解码器（与顶刊/基线统一：norm_first=True）====================
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_model, self.nhead, self.dim_feedforward, self.dropout,
            batch_first=True, activation='gelu', norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, self.num_decoder_layers)
        
        # ==================== 创新点B: 门控物理校正器 ====================
        if self.use_physics_corrector:
            # 解码状态决定当前时刻对运动学先验的信任程度。
            self.smoothness_gate = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 4),
                nn.Tanh(),
                nn.Linear(self.d_model // 4, 1),
                nn.Sigmoid()
            )
            self.physics_correction_scale = nn.Parameter(torch.tensor(0.0))
            self.physics_corrector_max_scale = 0.05
            self.physics_prior_pull_weight = 1.0
            self.physics_context_embedding = nn.Linear(self.output_dim, self.d_model)
            self.physics_context_scale = nn.Parameter(torch.tensor(0.0))
            self.physics_context_max_scale = 0.05
            
        # ==================== 创新点C: 多头轨迹解码器 ====================
        self.output_projection = nn.Linear(self.d_model, self.output_dim)
        if self.use_multi_head_output:
            self.trajectory_delta_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.GELU(),
                nn.Linear(self.d_model // 2, self.output_dim)
            )
            self.multihead_fusion_gate = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 4),
                nn.GELU(),
                nn.Linear(self.d_model // 4, self.output_dim),
                nn.Sigmoid()
            )
            self.trajectory_delta_scale = nn.Parameter(torch.tensor(0.0))
            self.trajectory_delta_max_scale = 0.05
            self.physics_fusion_gate = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 4),
                nn.GELU(),
                nn.Linear(self.d_model // 4, self.output_dim),
            )
            self.physics_prior_max_scale = 1.0
            self.physics_prior = None
            has_physical_scalers = (
                self.input_scaler_mean.numel() == self.input_dim
                and self.input_scaler_scale.numel() == self.input_dim
                and self.output_scaler_mean.numel() == self.output_dim
                and self.output_scaler_scale.numel() == self.output_dim
            )
            if has_physical_scalers:
                from .baseline_models import (
                    RotatingEarth3DOFBaseline,
                    SphericalKinematicBaseline,
                )

                prior_class = (
                    RotatingEarth3DOFBaseline
                    if self.prior_type == 'rotating_3dof'
                    else SphericalKinematicBaseline
                )
                prior_kwargs = {
                    'input_dim': self.input_dim,
                    'output_dim': self.output_dim,
                    'input_scaler_mean': self.input_scaler_mean,
                    'input_scaler_scale': self.input_scaler_scale,
                    'output_scaler_mean': self.output_scaler_mean,
                    'output_scaler_scale': self.output_scaler_scale,
                    'sampling_interval_s': self.sampling_interval_s,
                }
                if self.prior_type == 'rotating_3dof':
                    prior_kwargs['integration_stride'] = 8
                self.physics_prior = prior_class(**prior_kwargs)
        
        self._init_weights()
        if self.use_sparse_attention:
            # Module A acts in the primary encoder. The temporary standard
            # encoder exists only to provide matched initialization weights.
            self.transformer_encoder = nn.Identity()
    
    def _init_weights(self):
        """统一的权重初始化"""
        def with_module_seed(module_name, init_fn):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(_stable_module_seed(self._init_seed, module_name))
                init_fn()

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                def init_linear(module=module):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(module.bias, -bound, bound)
                with_module_seed(name, init_linear)
            elif isinstance(module, nn.MultiheadAttention):
                def init_attention(module=module):
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=1.0)
                    if module.in_proj_bias is not None:
                        nn.init.constant_(module.in_proj_bias, 0)
                    if module.bias_k is not None:
                        nn.init.xavier_uniform_(module.bias_k, gain=1.0)
                    if module.bias_v is not None:
                        nn.init.xavier_uniform_(module.bias_v, gain=1.0)
                with_module_seed(name, init_attention)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        # 创新点 A：物理感知注意力——偏置缩放初始化为小值，训练中逐步打开
        if self.use_sparse_attention:
            for std_layer, phys_layer in zip(self.transformer_encoder.layers, self.physics_encoder.layers):
                self._copy_standard_encoder_layer(std_layer, phys_layer)
                attn = phys_layer.self_attn
                nn.init.constant_(attn.bias_scale, -3.0)
                nn.init.constant_(attn.temporal_decay_raw, 0.0)

        if self.use_physics_corrector:
            gate_last = self.smoothness_gate[2]
            nn.init.constant_(gate_last.weight, 0.0)
            nn.init.constant_(gate_last.bias, -4.0)
            nn.init.constant_(self.physics_correction_scale, 0.0)
            nn.init.constant_(self.physics_context_scale, 0.0)
        
        if self.use_multi_head_output:
            delta_last = self.trajectory_delta_head[-1]
            nn.init.xavier_uniform_(delta_last.weight, gain=0.05)
            nn.init.constant_(delta_last.bias, 0.0)
            gate_last = self.multihead_fusion_gate[2]
            nn.init.constant_(gate_last.weight, 0.0)
            nn.init.constant_(gate_last.bias, -3.0)
            nn.init.constant_(self.trajectory_delta_scale, 0.0)
            physics_gate_last = self.physics_fusion_gate[2]
            nn.init.constant_(physics_gate_last.weight, 0.0)
            nn.init.constant_(physics_gate_last.bias, 0.0)

    @staticmethod
    def _copy_standard_encoder_layer(std_layer, phys_layer):
        """Initialize the physics-aware encoder from the standard encoder backbone."""
        with torch.no_grad():
            q_weight, k_weight, v_weight = std_layer.self_attn.in_proj_weight.chunk(3, dim=0)
            q_bias, k_bias, v_bias = std_layer.self_attn.in_proj_bias.chunk(3, dim=0)
            phys_layer.self_attn.q_proj.weight.copy_(q_weight)
            phys_layer.self_attn.k_proj.weight.copy_(k_weight)
            phys_layer.self_attn.v_proj.weight.copy_(v_weight)
            phys_layer.self_attn.q_proj.bias.copy_(q_bias)
            phys_layer.self_attn.k_proj.bias.copy_(k_bias)
            phys_layer.self_attn.v_proj.bias.copy_(v_bias)
            phys_layer.self_attn.out_proj.weight.copy_(std_layer.self_attn.out_proj.weight)
            phys_layer.self_attn.out_proj.bias.copy_(std_layer.self_attn.out_proj.bias)
            phys_layer.norm1.weight.copy_(std_layer.norm1.weight)
            phys_layer.norm1.bias.copy_(std_layer.norm1.bias)
            phys_layer.norm2.weight.copy_(std_layer.norm2.weight)
            phys_layer.norm2.bias.copy_(std_layer.norm2.bias)
            phys_layer.ffn[0].weight.copy_(std_layer.linear1.weight)
            phys_layer.ffn[0].bias.copy_(std_layer.linear1.bias)
            phys_layer.ffn[3].weight.copy_(std_layer.linear2.weight)
            phys_layer.ffn[3].bias.copy_(std_layer.linear2.bias)

    def _denormalize_source(self, src):
        if self.input_scaler_mean.numel() == self.input_dim and self.input_scaler_scale.numel() == self.input_dim:
            mean = self.input_scaler_mean.to(device=src.device, dtype=src.dtype)
            scale = self.input_scaler_scale.to(device=src.device, dtype=src.dtype)
            return src * scale + mean
        if self.require_physical_scaler and (self.use_sparse_attention or self.use_multi_head_output):
            raise RuntimeError("Formal physics modules require complete input scaler metadata.")
        return src

    def physics_gate_snapshot(self):
        """Return scalar gate diagnostics for audit logs and ablation figures."""
        snapshot = {
            "use_physics_attention": bool(self.use_sparse_attention),
            "use_physics_corrector": bool(self.use_physics_corrector),
            "use_multi_head_output": bool(self.use_multi_head_output),
            "use_prior_fusion": bool(self.use_multi_head_output and self.use_prior_fusion),
            "use_channel_residual": bool(
                self.use_multi_head_output and self.use_channel_residual
            ),
            "prior_type": self.prior_type,
            "prior_blend_mode": self.prior_blend_mode,
            "attention_residual_scale": 1.0 if self.use_sparse_attention else 0.0,
            "physics_correction_scale": float(
                self.physics_corrector_max_scale * torch.tanh(self.physics_correction_scale).detach().cpu()
            ) if self.use_physics_corrector else 0.0,
            "physics_context_scale": float(
                self.physics_context_max_scale * torch.tanh(self.physics_context_scale).detach().cpu()
            ) if self.use_physics_corrector else 0.0,
            "kinematic_prior_scale": float(
                self._nominal_prior_weight(256).mean().detach().cpu()
            ) if self.use_multi_head_output and self.use_prior_fusion else 0.0,
            "nominal_prior_weight_32": float(
                self._nominal_prior_weight(32).mean().detach().cpu()
            ) if self.use_multi_head_output and self.use_prior_fusion else 0.0,
            "nominal_prior_weight_128": float(
                self._nominal_prior_weight(128).mean().detach().cpu()
            ) if self.use_multi_head_output and self.use_prior_fusion else 0.0,
            "nominal_prior_weight_256": float(
                self._nominal_prior_weight(256).mean().detach().cpu()
            ) if self.use_multi_head_output and self.use_prior_fusion else 0.0,
            "physics_fusion_logit_max": float(
                self.physics_fusion_logit_max
            ) if self.use_multi_head_output and self.use_prior_fusion else 0.0,
            "trajectory_delta_scale": float(
                self.trajectory_delta_max_scale * torch.tanh(self.trajectory_delta_scale).detach().cpu()
            ) if self.use_multi_head_output and self.use_channel_residual else 0.0,
            "prior_mask": list(self.attention_prior_mask),
            "layers": [],
        }
        if self.use_sparse_attention:
            for layer in self.physics_encoder.layers:
                attention = layer.self_attn
                active_mask = torch.as_tensor(
                    attention.prior_mask,
                    device=attention.bias_scale.device,
                    dtype=attention.bias_scale.dtype,
                )
                scales = torch.sigmoid(attention.bias_scale) * active_mask
                diagnostics = {
                    "temporal_gate": scales[0],
                    "phase_gate": scales[1],
                    "geometry_gate": scales[2],
                    "temporal_decay": F.softplus(attention.temporal_decay_raw).mean(),
                    "phase_bandwidth": F.softplus(attention.phase_bandwidth_raw).mean(),
                    "geometry_bandwidth": F.softplus(attention.geo_bandwidth_raw).mean(),
                }
                snapshot["layers"].append({
                    key: float(value.detach().cpu()) for key, value in diagnostics.items()
                })
        if snapshot["layers"]:
            for key in snapshot["layers"][0]:
                snapshot[f"mean_{key}"] = float(np.mean([item[key] for item in snapshot["layers"]]))
        return snapshot

    def _denormalize_output(self, value):
        if self.output_scaler_mean.numel() == self.output_dim and self.output_scaler_scale.numel() == self.output_dim:
            mean = self.output_scaler_mean.to(device=value.device, dtype=value.dtype)
            scale = self.output_scaler_scale.to(device=value.device, dtype=value.dtype)
            return value * scale + mean
        return value

    def _normalize_output(self, value):
        if self.output_scaler_mean.numel() == self.output_dim and self.output_scaler_scale.numel() == self.output_dim:
            mean = self.output_scaler_mean.to(device=value.device, dtype=value.dtype)
            scale = self.output_scaler_scale.to(device=value.device, dtype=value.dtype).clamp_min(1e-12)
            return (value - mean) / scale
        return value

    def _build_kinematic_prior(
        self,
        tgt,
        source_physical=None,
        src=None,
        physics_prior_override=None,
    ):
        """Build a frozen-state 3-DOF kinematic prior for future decoder slots."""
        batch_size, seq_len, _ = tgt.shape
        device = tgt.device
        dtype = tgt.dtype
        row_is_context = tgt.abs().sum(dim=-1) > 1e-8
        context_lengths = row_is_context.long().sum(dim=1).clamp(min=1, max=seq_len)
        batch_idx = torch.arange(batch_size, device=device)
        last_idx = context_lengths - 1
        steps = torch.arange(seq_len, device=device, dtype=dtype).view(1, seq_len, 1)
        steps = torch.clamp(steps - last_idx.to(dtype).view(batch_size, 1, 1), min=0.0)

        has_physical_scalers = (
            self.output_scaler_mean.numel() == self.output_dim
            and self.output_scaler_scale.numel() == self.output_dim
        )
        if self.use_multi_head_output and self.physics_prior is not None and src is not None:
            future_lengths = seq_len - context_lengths
            if not torch.equal(future_lengths, future_lengths[:1].expand_as(future_lengths)):
                raise ValueError("Physics-prior fusion requires a common decoder context length per batch.")
            future_length = int(future_lengths[0].item())
            if future_length > 0:
                if physics_prior_override is not None:
                    if physics_prior_override.ndim != 3:
                        raise ValueError("physics_prior_override must have [batch, horizon, output] shape.")
                    expected = (batch_size, future_length, self.output_dim)
                    if tuple(physics_prior_override.shape) != expected:
                        raise ValueError(
                            "physics_prior_override shape mismatch: "
                            f"expected {expected}, got {tuple(physics_prior_override.shape)}."
                        )
                    physics_future = physics_prior_override.to(device=device, dtype=dtype)
                else:
                    with torch.no_grad(), torch.autocast(device_type=src.device.type, enabled=False):
                        physics_future = self.physics_prior(
                            src.float(), target_length=future_length
                        ).to(device=device, dtype=dtype)
                prior = tgt.clone()
                context_length = int(context_lengths[0].item())
                prior[:, context_length:context_length + future_length, :] = physics_future
                future_mask = (~row_is_context).to(dtype).unsqueeze(-1)
                return prior, future_mask

        if source_physical is not None and source_physical.size(-1) >= 6 and has_physical_scalers:
            state = source_physical[:, -1, :6]
            radius, longitude, latitude, speed, flight_path, heading = state.unbind(dim=-1)
            cos_latitude = torch.cos(latitude).abs().clamp_min(1e-8)
            dt_steps = steps * float(self.sampling_interval_s)
            dr_dt = speed * torch.sin(flight_path)
            dlongitude_dt = (
                speed * torch.cos(flight_path) * torch.sin(heading) / (radius * cos_latitude)
            )
            dlatitude_dt = speed * torch.cos(flight_path) * torch.cos(heading) / radius
            extrapolated_physical = torch.stack(
                (
                    radius.unsqueeze(1) + dt_steps[..., 0] * dr_dt.unsqueeze(1),
                    longitude.unsqueeze(1) + dt_steps[..., 0] * dlongitude_dt.unsqueeze(1),
                    latitude.unsqueeze(1) + dt_steps[..., 0] * dlatitude_dt.unsqueeze(1),
                ),
                dim=-1,
            )
            extrapolated_physical[..., 1] = torch.atan2(
                torch.sin(extrapolated_physical[..., 1]),
                torch.cos(extrapolated_physical[..., 1]),
            )
            extrapolated = self._normalize_output(extrapolated_physical)
        else:
            prev_idx = (context_lengths - 2).clamp(min=0)
            last_pos = tgt[batch_idx, last_idx, :]
            prev_pos = tgt[batch_idx, prev_idx, :]
            velocity = torch.clamp(
                last_pos - prev_pos,
                min=-float(self.kinematic_velocity_clip),
                max=float(self.kinematic_velocity_clip),
            )
            velocity = torch.where((context_lengths > 1).view(batch_size, 1), velocity, torch.zeros_like(velocity))
            extrapolated = last_pos.unsqueeze(1) + steps * velocity.unsqueeze(1)

        prior = torch.where(row_is_context.unsqueeze(-1), tgt, extrapolated)
        future_mask = (~row_is_context).to(dtype).unsqueeze(-1)
        return prior, future_mask

    def _confidence_weighted_prior_pull(self, kinematic_prior, output, future_mask):
        """Return a conservative prior pull that fades when the prior disagrees."""
        raw_delta = kinematic_prior - output
        disagreement = raw_delta.abs().mean(dim=-1, keepdim=True).clamp(max=8.0)
        confidence = torch.exp(-float(self.physics_prior_confidence_sharpness) * disagreement)
        return future_mask * confidence * torch.clamp(raw_delta, min=-1.0, max=1.0)

    def _nominal_prior_weight(self, horizon_steps):
        """Return the zero-disagreement prior weight at a forecast horizon."""
        bias = self.physics_fusion_gate[2].bias
        steps = torch.as_tensor(
            horizon_steps,
            device=bias.device,
            dtype=bias.dtype,
        )
        time_constant_steps = max(
            float(self.physics_prior_time_constant_s) / float(self.sampling_interval_s),
            1.0,
        )
        scheduled_confidence = torch.exp(
            -torch.pow(steps / time_constant_steps, float(self.physics_prior_decay_power))
        ).clamp(0.01, 0.99)
        schedule_logit = torch.logit(scheduled_confidence)
        return torch.sigmoid(schedule_logit + bias)

    def _scheduled_prior_weight(self, kinematic_prior, output, decoder_output, future_mask):
        """Blend a short-horizon dynamics prior with a learned long-horizon forecast."""
        raw_delta = kinematic_prior - output
        disagreement = raw_delta.abs().mean(dim=-1, keepdim=True).clamp(max=8.0)
        step_index = torch.arange(
            1,
            future_mask.size(1) + 1,
            device=future_mask.device,
            dtype=future_mask.dtype,
        ).view(1, -1, 1)
        context_length = future_mask.size(1) - future_mask.sum(dim=1, keepdim=True)
        future_steps = (
            (step_index - context_length).clamp_min(0.0)
            * future_mask
            * float(self.sampling_interval_s)
        )
        scheduled_confidence = torch.exp(
            -torch.pow(
                future_steps / float(self.physics_prior_time_constant_s),
                float(self.physics_prior_decay_power),
            )
        ).clamp(0.01, 0.99)
        if self.prior_blend_mode == 'schedule_only':
            return future_mask * scheduled_confidence
        schedule_logit = torch.logit(scheduled_confidence)
        learned_logit = float(self.physics_fusion_logit_max) * torch.tanh(
            self.physics_fusion_gate(decoder_output)
        )
        prior_weight = torch.sigmoid(
            schedule_logit
            + learned_logit
            - float(self.physics_prior_disagreement_sharpness) * disagreement
        )
        return future_mask * prior_weight

    def _apply_physics_correction(
        self,
        output,
        decoder_output,
        kinematic_prior,
        future_mask,
    ):
        """Apply a bounded, confidence-weighted kinematic residual correction."""
        gate = self.smoothness_gate(decoder_output)
        correction_scale = self.physics_corrector_max_scale * torch.tanh(
            self.physics_correction_scale
        )
        prior_pull = self._confidence_weighted_prior_pull(
            kinematic_prior,
            output,
            future_mask,
        )
        residual = gate * correction_scale * self.physics_prior_pull_weight * prior_pull
        return output + residual
    
    def forward(self, src, tgt, tgt_mask=None, physics_prior_override=None):
        """PLGAFormer前向传播"""
        # 1. 输入嵌入 + 位置编码
        source_physical = self._denormalize_source(src)
        src_embedded = self.input_embedding(src)
        src_embedded = self.pos_encoder(src_embedded)
        
        tgt_embedded = self.output_embedding(tgt)
        kinematic_prior = None
        future_mask = None
        if self.use_physics_corrector:
            kinematic_prior, future_mask = self._build_kinematic_prior(
                tgt,
                source_physical,
                src=src,
                physics_prior_override=physics_prior_override,
            )
            physics_hint = future_mask * torch.clamp(kinematic_prior - tgt, min=-1.0, max=1.0)
            context_scale = self.physics_context_max_scale * torch.tanh(self.physics_context_scale)
            tgt_embedded = tgt_embedded + context_scale * self.physics_context_embedding(physics_hint)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        
        # 2. 编码器
        standard_encoder_output = self.transformer_encoder(src_embedded)
        if self.use_sparse_attention:
            # 创新点 A：物理感知注意力作为标准编码器上的有界残差，
            # 训练初期可退化为 Transformer baseline，避免整体替换带来的不稳定。
            encoder_output = self.physics_encoder(src_embedded, raw_input=source_physical)
        else:
            encoder_output = standard_encoder_output
        
        # 3. 解码器
        decoder_output = self.transformer_decoder(tgt_embedded, encoder_output, tgt_mask=tgt_mask)
        
        # 4. 创新点C：多头输出
        if self.use_multi_head_output:
            base_output = self.output_projection(decoder_output)
            if self.use_prior_fusion:
                if kinematic_prior is None or future_mask is None:
                    kinematic_prior, future_mask = self._build_kinematic_prior(
                        tgt,
                        source_physical,
                        src=src,
                        physics_prior_override=physics_prior_override,
                    )
                prior_weight = self._scheduled_prior_weight(
                    kinematic_prior,
                    base_output,
                    decoder_output,
                    future_mask,
                )
                base_output = base_output + prior_weight * (kinematic_prior - base_output)
            output = base_output
            if self.use_channel_residual:
                if future_mask is None:
                    future_mask = (tgt.abs().sum(dim=-1) <= 1e-8).to(tgt.dtype).unsqueeze(-1)
                delta = torch.tanh(self.trajectory_delta_head(decoder_output))
                gate = self.multihead_fusion_gate(decoder_output)
                delta_scale = self.trajectory_delta_max_scale * torch.tanh(
                    self.trajectory_delta_scale
                )
                output = output + future_mask * gate * delta_scale * delta
        else:
            output = self.output_projection(decoder_output)

        # 5. 创新点B：输出空间门控运动学先验残差
        if self.use_physics_corrector:
            if kinematic_prior is None or future_mask is None:
                kinematic_prior, future_mask = self._build_kinematic_prior(
                    tgt,
                    source_physical,
                    src=src,
                    physics_prior_override=physics_prior_override,
                )
            output = self._apply_physics_correction(
                output,
                decoder_output,
                kinematic_prior,
                future_mask,
            )
        
        return output


# 为了兼容性，创建PLGAFormer别名
def PLGAFormer(**kwargs):
    """PLGAFormer模型创建函数 - 统一配置"""
    if 'input_dim' not in kwargs:
        kwargs['input_dim'] = 6
    
    # 使用保守配置作为默认值 - 适合HGV轨迹预测
    model_kwargs = {
        'input_dim': kwargs.get('input_dim', 6),
        'd_model': kwargs.get('d_model', 128),              # 保守配置：降低复杂度
        'nhead': kwargs.get('nhead', 4),                    # 保守配置：4个头  
        'num_encoder_layers': kwargs.get('num_encoder_layers', 3),  # 保守配置：3层编码器
        'num_decoder_layers': kwargs.get('num_decoder_layers', 2),  # 保守配置：2层解码器
        'dim_feedforward': kwargs.get('dim_feedforward', 512),      # 保守配置：4*d_model
        'dropout': kwargs.get('dropout', 0.1),                     # 保守配置：标准dropout
        'use_sparse_attention': kwargs.get('use_sparse_attention', False),
        'use_physics_corrector': kwargs.get('use_physics_corrector', False),
        'use_multi_head_output': kwargs.get('use_multi_head_output', False),
        'use_adaptive_fusion': kwargs.get('use_adaptive_fusion', True),
        'use_prior_fusion': kwargs.get('use_prior_fusion', True),
        'use_channel_residual': kwargs.get('use_channel_residual', False),
        'prior_type': kwargs.get('prior_type', 'rotating_3dof'),
        'prior_blend_mode': kwargs.get('prior_blend_mode', 'adaptive'),
    }
    
    return PLGAFormerTransformer(**model_kwargs)
