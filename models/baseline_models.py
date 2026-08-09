#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线模型定义

包含用于对比实验的基础模型：
1. StandardTransformer - 标准Transformer基线模型
2. KalmanFilter - 卡尔曼滤波器简单基线模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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


class KalmanFilter(nn.Module):
    """
    固定参数卡尔曼滤波器基线模型（顶刊常见做法）
    
    实现简单的线性卡尔曼滤波器用于轨迹预测，参数固定不训练。
    适用于HGV轨迹预测任务的简单基线对比，无需训练即可使用。
    """
    
    def __init__(self, input_dim=6, output_dim=3, **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 状态维度：位置(h, λ, φ) + 速度(vh, vλ, vφ) + 速度大小V = 7维
        self.state_dim = 7
        
        # 固定参数：状态转移矩阵 F（单位矩阵 + 小扰动，模拟匀速运动）
        F_fixed = torch.eye(self.state_dim)
        # 添加小的速度项（对角线上方一行）
        F_fixed[0:3, 3:6] = torch.eye(3) * 0.1  # 位置受速度影响
        self.register_buffer('F', F_fixed)
        
        # 固定参数：观测矩阵 H（从7维状态映射到6维观测）
        # 假设观测到位置和速度的前3维（简化设计）
        H_fixed = torch.zeros(self.input_dim, self.state_dim)
        H_fixed[:3, :3] = torch.eye(3)  # 观测位置
        if self.input_dim >= 6:
            H_fixed[3:6, 3:6] = torch.eye(3) * 0.5  # 观测速度（权重0.5）
        self.register_buffer('H', H_fixed)
        
        # 固定参数：过程噪声协方差 Q（对角矩阵，小噪声）
        Q_diag = torch.ones(self.state_dim) * 0.01
        self.register_buffer('Q_diag', Q_diag)
        
        # 固定参数：观测噪声协方差 R（对角矩阵）
        R_diag = torch.ones(self.input_dim) * 0.1
        self.register_buffer('R_diag', R_diag)
        
        # 固定参数：输出映射矩阵（从7维状态映射到3维输出 [r, λ, φ]）
        output_mapping_fixed = torch.zeros(self.state_dim, self.output_dim)
        output_mapping_fixed[:3, :3] = torch.eye(3)  # 直接输出位置
        self.register_buffer('output_mapping', output_mapping_fixed)
        
        # 固定参数：初始状态协方差
        P_init_diag = torch.ones(self.state_dim) * 1.0
        self.register_buffer('P_init_diag', P_init_diag)
        
    def forward(self, src, tgt=None, tgt_mask=None):
        """
        Args:
            src: 输入序列 [batch_size, seq_len, input_dim]
            tgt: 目标序列 [batch_size, pred_len, output_dim] (训练时使用)
            tgt_mask: 目标掩码 (未使用)
        
        Returns:
            预测序列 [batch_size, pred_len, output_dim]
        """
        batch_size, seq_len, _ = src.shape
        device = src.device
        
        if tgt is not None:
            pred_len = tgt.shape[1]
        else:
            # 使用HGVConfig统一配置获取预测长度
            from . import HGVConfig
            train_config = HGVConfig.get_train_config()
            pred_len = train_config['pred_len']
        
        # 初始化状态和协方差（使用固定参数）
        x = torch.zeros(batch_size, self.state_dim, device=device)
        P = torch.diag_embed(self.P_init_diag.unsqueeze(0).expand(batch_size, -1))
        
        # 构建协方差矩阵（固定参数）
        Q = torch.diag_embed(self.Q_diag.unsqueeze(0).expand(batch_size, -1))  # [batch_size, state_dim, state_dim]
        R = torch.diag_embed(self.R_diag.unsqueeze(0).expand(batch_size, -1))  # [batch_size, input_dim, input_dim]
        
        # 使用输入序列更新状态估计
        for t in range(seq_len):
            # 预测步骤
            x = torch.matmul(x.unsqueeze(1), self.F.unsqueeze(0)).squeeze(1)
            F_batch = self.F.unsqueeze(0).expand(batch_size, -1, -1)
            P = torch.matmul(torch.matmul(F_batch, P), F_batch.transpose(1, 2)) + Q
            
            # 更新步骤
            z = src[:, t, :]  # 当前观测
            
            # 计算卡尔曼增益
            # H矩阵维度: [6, 7] (观测维度 x 状态维度)
            # H_batch维度: [batch_size, 6, 7]
            H_batch = self.H.unsqueeze(0).expand(batch_size, -1, -1)
            # S = H @ P @ H^T + R
            # H @ P: [32, 6, 7] @ [32, 7, 7] = [32, 6, 7]
            # (H @ P) @ H^T: [32, 6, 7] @ [32, 7, 6] = [32, 6, 6]
            S = torch.matmul(torch.matmul(H_batch, P), H_batch.transpose(1, 2)) + R
            
            # 使用Cholesky分解提高数值稳定性
            try:
                L = torch.linalg.cholesky(S)
                S_inv = torch.cholesky_inverse(L)
            except RuntimeError:
                # 如果Cholesky分解失败，添加正则化项并使用伪逆
                S_reg = S + torch.eye(self.input_dim, device=device).unsqueeze(0) * 1e-4
                S_inv = torch.pinverse(S_reg)
            
            # 卡尔曼增益 K = P @ H^T @ S^(-1)
            # P @ H^T: [32, 7, 7] @ [32, 7, 6] = [32, 7, 6]
            # (P @ H^T) @ S^(-1): [32, 7, 6] @ [32, 6, 6] = [32, 7, 6]
            K = torch.matmul(torch.matmul(P, H_batch.transpose(1, 2)), S_inv)
            
            # 状态更新
            # 观测预测: H @ x = [32, 6, 7] @ [32, 7, 1] = [32, 6, 1]
            y = z.unsqueeze(2) - torch.matmul(H_batch, x.unsqueeze(2))
            x = x + torch.matmul(K, y).squeeze(2)
            
            # 协方差更新 P = (I - K @ H) @ P
            # K @ H: [32, 7, 6] @ [32, 6, 7] = [32, 7, 7]
            I_KH = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1) - torch.matmul(K, H_batch)
            P = torch.matmul(I_KH, P)
        
        # 预测未来状态
        predictions = []
        current_x = x.clone()
        
        for t in range(pred_len):
            # 预测下一个状态
            F_batch = self.F.unsqueeze(0).expand(batch_size, -1, -1)
            current_x = torch.matmul(current_x.unsqueeze(1), F_batch).squeeze(1)
            
            # 映射到输出空间
            pred = torch.matmul(current_x, self.output_mapping)
            predictions.append(pred)
        
        # 堆叠预测结果
        predictions = torch.stack(predictions, dim=1)  # [batch_size, pred_len, output_dim]
        
        return predictions


class SphericalKinematicBaseline(nn.Module):
    """Parameter-free frozen-state baseline in spherical coordinates.

    The last observed speed, flight-path angle, and heading are held constant.
    This is an analytical kinematic comparator, not a numerical integration of
    the full HGV equations of motion.
    """

    def __init__(
        self,
        input_dim=6,
        output_dim=3,
        input_scaler_mean=None,
        input_scaler_scale=None,
        output_scaler_mean=None,
        output_scaler_scale=None,
        sampling_interval_s=1.0,
        **kwargs,
    ):
        super().__init__()
        if input_dim < 6 or output_dim != 3:
            raise ValueError("SphericalKinematicBaseline requires 6-D input and 3-D position output.")
        arrays = (input_scaler_mean, input_scaler_scale, output_scaler_mean, output_scaler_scale)
        if any(value is None for value in arrays):
            raise ValueError("Physical scaler metadata is required for the kinematic baseline.")
        self.register_buffer("input_scaler_mean", torch.as_tensor(input_scaler_mean, dtype=torch.float32))
        self.register_buffer("input_scaler_scale", torch.as_tensor(input_scaler_scale, dtype=torch.float32))
        self.register_buffer("output_scaler_mean", torch.as_tensor(output_scaler_mean, dtype=torch.float32))
        self.register_buffer("output_scaler_scale", torch.as_tensor(output_scaler_scale, dtype=torch.float32))
        if self.input_scaler_mean.numel() != 6 or self.output_scaler_mean.numel() != 3:
            raise ValueError("Unexpected scaler dimensions for the kinematic baseline.")
        self.sampling_interval_s = float(sampling_interval_s)

    def forward(self, src, tgt=None, tgt_mask=None, target_length=None, decoder_context=None):
        if target_length is None:
            if tgt is None:
                raise ValueError("target_length or tgt is required.")
            target_length = tgt.size(1)
        target_length = int(target_length)
        if target_length < 1:
            raise ValueError("target_length must be positive.")

        mean = self.input_scaler_mean.to(device=src.device, dtype=src.dtype)
        scale = self.input_scaler_scale.to(device=src.device, dtype=src.dtype)
        state = src[:, -1, :6] * scale + mean
        radius, longitude, latitude, speed, flight_path, heading = state.unbind(dim=-1)
        cos_latitude = torch.cos(latitude).abs().clamp_min(1e-8)
        dr_dt = speed * torch.sin(flight_path)
        dlongitude_dt = speed * torch.cos(flight_path) * torch.sin(heading) / (radius * cos_latitude)
        dlatitude_dt = speed * torch.cos(flight_path) * torch.cos(heading) / radius
        elapsed = (
            torch.arange(1, target_length + 1, device=src.device, dtype=src.dtype)
            * self.sampling_interval_s
        ).view(1, target_length)
        physical = torch.stack(
            (
                radius.unsqueeze(1) + elapsed * dr_dt.unsqueeze(1),
                longitude.unsqueeze(1) + elapsed * dlongitude_dt.unsqueeze(1),
                latitude.unsqueeze(1) + elapsed * dlatitude_dt.unsqueeze(1),
            ),
            dim=-1,
        )
        physical[..., 1] = torch.atan2(torch.sin(physical[..., 1]), torch.cos(physical[..., 1]))
        output_mean = self.output_scaler_mean.to(device=src.device, dtype=src.dtype)
        output_scale = self.output_scaler_scale.to(device=src.device, dtype=src.dtype)
        return (physical - output_mean) / output_scale


class RotatingEarth3DOFBaseline(nn.Module):
    """Parameter-free rotating-Earth 3-DOF propagation baseline.

    Equivalent lift/drag coefficients and bank angle are identified from the
    final observed state differences. They are then held fixed during an RK4
    rollout of the same rotating spherical-Earth point-mass equations used by
    the simulator. This is a transparent numerical physics baseline, not a
    claim to reproduce a published analytical predictor.
    """

    def __init__(
        self,
        input_dim=6,
        output_dim=3,
        input_scaler_mean=None,
        input_scaler_scale=None,
        output_scaler_mean=None,
        output_scaler_scale=None,
        sampling_interval_s=1.0,
        identification_steps=16,
        integration_stride=8,
        **kwargs,
    ):
        super().__init__()
        if input_dim < 6 or output_dim != 3:
            raise ValueError("RotatingEarth3DOFBaseline requires 6-D state input.")
        arrays = (input_scaler_mean, input_scaler_scale, output_scaler_mean, output_scaler_scale)
        if any(value is None for value in arrays):
            raise ValueError("Physical scaler metadata is required for the 3-DOF baseline.")
        self.register_buffer("input_scaler_mean", torch.as_tensor(input_scaler_mean, dtype=torch.float32))
        self.register_buffer("input_scaler_scale", torch.as_tensor(input_scaler_scale, dtype=torch.float32))
        self.register_buffer("output_scaler_mean", torch.as_tensor(output_scaler_mean, dtype=torch.float32))
        self.register_buffer("output_scaler_scale", torch.as_tensor(output_scaler_scale, dtype=torch.float32))
        self.sampling_interval_s = float(sampling_interval_s)
        self.identification_steps = max(2, int(identification_steps))
        self.integration_stride = max(1, int(integration_stride))
        self.earth_radius = 6_378_000.0
        self.mu = 3.986004418e14
        self.omega = 7.292115e-5
        self.mass = 907.2
        self.reference_area = 0.4839
        # These atmosphere tables are immutable model constants. Registering
        # them once avoids rebuilding and transferring four tensors for every
        # RK4 dynamics evaluation (128 evaluations for a 256-step rollout at
        # the default integration stride).
        self.register_buffer(
            "_atmosphere_base_h",
            torch.tensor(
                [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_atmosphere_base_t",
            torch.tensor(
                [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_atmosphere_base_p",
            torch.tensor(
                [101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420, 0.3734],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_atmosphere_lapse",
            torch.tensor(
                [-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @staticmethod
    def _safe_signed(value, floor=1e-8):
        sign = torch.where(value >= 0.0, torch.ones_like(value), -torch.ones_like(value))
        return sign * value.abs().clamp_min(float(floor))

    @staticmethod
    def _wrapped_difference(value):
        delta = value[:, 1:] - value[:, :-1]
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    def _atmosphere(self, altitude):
        dtype, device = altitude.dtype, altitude.device
        base_h = self._atmosphere_base_h.to(device=device, dtype=dtype)
        base_t = self._atmosphere_base_t.to(device=device, dtype=dtype)
        base_p = self._atmosphere_base_p.to(device=device, dtype=dtype)
        lapse = self._atmosphere_lapse.to(device=device, dtype=dtype)
        h = altitude.clamp_min(0.0)
        layer = torch.bucketize(h, base_h[1:-1]).clamp(max=len(lapse) - 1)
        h0 = base_h[layer]
        t0 = base_t[layer]
        p0 = base_p[layer]
        rate = lapse[layer]
        gas_constant = 287.05287
        gravity = 9.80665
        temperature = t0 + rate * (h - h0)
        pressure_gradient = p0 * torch.pow(
            t0 / temperature.clamp_min(1.0), gravity / (gas_constant * self._safe_signed(rate, 1e-12))
        )
        pressure_isothermal = p0 * torch.exp(-gravity * (h - h0) / (gas_constant * t0))
        pressure = torch.where(rate.abs() < 1e-12, pressure_isothermal, pressure_gradient)
        above = h >= base_h[-1]
        pressure_above = base_p[-1] * torch.exp(
            -gravity * (h - base_h[-1]) / (gas_constant * base_t[-1])
        )
        pressure = torch.where(above, pressure_above, pressure)
        temperature = torch.where(above, base_t[-1], temperature)
        return pressure / (gas_constant * temperature)

    def _rotation_terms(self, state):
        radius, _, latitude, speed, gamma, heading = state.unbind(dim=-1)
        sin_gamma, cos_gamma = torch.sin(gamma), torch.cos(gamma)
        sin_heading, cos_heading = torch.sin(heading), torch.cos(heading)
        sin_lat, cos_lat = torch.sin(latitude), torch.cos(latitude)
        safe_speed = speed.clamp_min(1.0)
        safe_cos_gamma = self._safe_signed(cos_gamma)
        velocity_term = self.omega**2 * radius * cos_lat * (
            sin_gamma * cos_lat - cos_gamma * sin_lat * cos_heading
        )
        gamma_term = (
            2.0 * self.omega * cos_lat * sin_heading
            + self.omega**2 * radius * cos_lat / safe_speed
            * (cos_gamma * cos_lat + sin_gamma * sin_lat * cos_heading)
        )
        heading_term = (
            2.0 * self.omega * sin_lat
            + self.omega**2 * radius * sin_heading * sin_lat * cos_lat
            / (safe_speed * safe_cos_gamma)
            - 2.0 * self.omega * cos_lat * cos_heading * torch.tan(gamma)
        )
        return velocity_term, gamma_term, heading_term

    def _identify_controls(self, history):
        dt = self.sampling_interval_s
        diffs = history[:, 1:] - history[:, :-1]
        diffs[..., 1] = self._wrapped_difference(history[..., 1])
        diffs[..., 5] = self._wrapped_difference(history[..., 5])
        # Averaging wrapped finite differences is the deterministic secant
        # slope over the identification window and avoids CUDA median kernels.
        derivative = torch.mean(diffs / dt, dim=1)
        state = history[:, -1]
        radius, _, latitude, speed, gamma, heading = state.unbind(dim=-1)
        gravity = self.mu / radius.square()
        velocity_rotation, gamma_rotation, heading_rotation = self._rotation_terms(state)
        drag = -self.mass * (derivative[:, 3] + gravity * torch.sin(gamma) - velocity_rotation)
        gamma_base = -(gravity / speed - speed / radius) * torch.cos(gamma) + gamma_rotation
        heading_base = (
            speed * torch.cos(gamma) * torch.sin(heading) * torch.tan(latitude) / radius
            + heading_rotation
        )
        lift_cos_bank = self.mass * speed * (derivative[:, 4] - gamma_base)
        lift_sin_bank = (
            self.mass * speed * self._safe_signed(torch.cos(gamma))
            * (derivative[:, 5] - heading_base)
        )
        lift = torch.sqrt(lift_cos_bank.square() + lift_sin_bank.square())
        bank = torch.atan2(lift_sin_bank, lift_cos_bank)
        density = self._atmosphere(radius - self.earth_radius)
        dynamic_area = 0.5 * density * speed.square() * self.reference_area
        # Explicit admissibility bounds protect inverse identification from
        # sensor-difference noise; these are reported as part of the baseline.
        cl = (lift / dynamic_area.clamp_min(1e-6)).clamp(0.0, 2.0)
        cd = (drag / dynamic_area.clamp_min(1e-6)).clamp(1e-4, 2.0)
        return cl, cd, bank

    def _dynamics(self, state, cl, cd, bank):
        radius, _, latitude, speed, gamma, heading = state.unbind(dim=-1)
        safe_speed = speed.clamp_min(1.0)
        safe_cos_lat = self._safe_signed(torch.cos(latitude))
        safe_cos_gamma = self._safe_signed(torch.cos(gamma))
        density = self._atmosphere(radius - self.earth_radius)
        dynamic_pressure = 0.5 * density * speed.square()
        lift = cl * dynamic_pressure * self.reference_area
        drag = cd * dynamic_pressure * self.reference_area
        gravity = self.mu / radius.square()
        velocity_rotation, gamma_rotation, heading_rotation = self._rotation_terms(state)
        return torch.stack(
            [
                speed * torch.sin(gamma),
                speed * torch.cos(gamma) * torch.sin(heading) / (radius * safe_cos_lat),
                speed * torch.cos(gamma) * torch.cos(heading) / radius,
                -drag / self.mass - gravity * torch.sin(gamma) + velocity_rotation,
                lift * torch.cos(bank) / (self.mass * safe_speed)
                - (gravity / safe_speed - safe_speed / radius) * torch.cos(gamma)
                + gamma_rotation,
                lift * torch.sin(bank) / (self.mass * safe_speed * safe_cos_gamma)
                + safe_speed * torch.cos(gamma) * torch.sin(heading) * torch.tan(latitude) / radius
                + heading_rotation,
            ],
            dim=-1,
        )

    def forward(self, src, tgt=None, tgt_mask=None, target_length=None, decoder_context=None):
        del tgt_mask, decoder_context
        if target_length is None:
            if tgt is None:
                raise ValueError("target_length or tgt is required.")
            target_length = tgt.size(1)
        target_length = int(target_length)
        mean = self.input_scaler_mean.to(device=src.device, dtype=src.dtype)
        scale = self.input_scaler_scale.to(device=src.device, dtype=src.dtype)
        history = src[:, -min(self.identification_steps + 1, src.size(1)):, :6] * scale + mean
        cl, cd, bank = self._identify_controls(history)
        state = history[:, -1]
        output_segments = []
        base_dt = self.sampling_interval_s
        completed_steps = 0
        while completed_steps < target_length:
            segment_steps = min(self.integration_stride, target_length - completed_steps)
            dt = base_dt * float(segment_steps)
            start_state = state
            k1 = self._dynamics(start_state, cl, cd, bank)
            k2 = self._dynamics(start_state + 0.5 * dt * k1, cl, cd, bank)
            k3 = self._dynamics(start_state + 0.5 * dt * k2, cl, cd, bank)
            k4 = self._dynamics(start_state + dt * k3, cl, cd, bank)
            state = start_state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            state = torch.cat(
                [
                    state[:, :1],
                    torch.atan2(torch.sin(state[:, 1:2]), torch.cos(state[:, 1:2])),
                    state[:, 2:5],
                    torch.atan2(torch.sin(state[:, 5:6]), torch.cos(state[:, 5:6])),
                ],
                dim=-1,
            )
            delta = state - start_state
            delta = torch.cat(
                [
                    delta[:, :1],
                    torch.atan2(torch.sin(delta[:, 1:2]), torch.cos(delta[:, 1:2])),
                    delta[:, 2:5],
                    torch.atan2(torch.sin(delta[:, 5:6]), torch.cos(delta[:, 5:6])),
                ],
                dim=-1,
            )
            fractions = (
                torch.arange(1, segment_steps + 1, device=state.device, dtype=state.dtype)
                / float(segment_steps)
            ).view(1, segment_steps, 1)
            interpolated = start_state.unsqueeze(1) + fractions * delta.unsqueeze(1)
            output_segments.append(interpolated[..., :3])
            completed_steps += segment_steps
        physical = torch.cat(output_segments, dim=1)
        # A single synchronized check preserves the numerical guard without
        # forcing a CPU/GPU barrier after every integration segment.
        if not torch.isfinite(physical).all():
            raise FloatingPointError("Non-finite rotating-Earth 3-DOF rollout.")
        output_mean = self.output_scaler_mean.to(device=src.device, dtype=src.dtype)
        output_scale = self.output_scaler_scale.to(device=src.device, dtype=src.dtype)
        return (physical - output_mean) / output_scale


class DLinear(nn.Module):
    """Channel-wise decomposition-linear forecasting baseline."""

    def __init__(self, input_dim=6, output_dim=3, seq_len=None, pred_len=None, moving_avg=25, **kwargs):
        super().__init__()
        from . import HGVConfig

        train_config = HGVConfig.get_train_config()
        self.seq_len = int(seq_len or train_config['seq_len'])
        self.pred_len = int(pred_len or train_config['pred_len'])
        self.output_dim = int(output_dim)
        self.moving_avg = int(moving_avg)
        if self.moving_avg < 1 or self.moving_avg % 2 == 0:
            raise ValueError("moving_avg must be a positive odd integer.")
        self.seasonal_heads = nn.ModuleList(
            [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.output_dim)]
        )
        self.trend_heads = nn.ModuleList(
            [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.output_dim)]
        )

    def _moving_average(self, values):
        padding = (self.moving_avg - 1) // 2
        front = values[..., :1].expand(*values.shape[:-1], padding)
        end = values[..., -1:].expand(*values.shape[:-1], padding)
        padded = torch.cat([front, values, end], dim=-1)
        return F.avg_pool1d(padded, kernel_size=self.moving_avg, stride=1)

    def forward(self, src, tgt=None, tgt_mask=None, target_length=None, decoder_context=None):
        positions = src[..., :self.output_dim].transpose(1, 2)
        if positions.size(-1) != self.seq_len:
            positions = F.interpolate(positions, size=self.seq_len, mode='linear', align_corners=False)
        trend = self._moving_average(positions)
        seasonal = positions - trend
        channels = [
            self.seasonal_heads[index](seasonal[:, index, :])
            + self.trend_heads[index](trend[:, index, :])
            for index in range(self.output_dim)
        ]
        output = torch.stack(channels, dim=-1)
        if target_length is not None and int(target_length) != self.pred_len:
            output = F.interpolate(
                output.transpose(1, 2),
                size=int(target_length),
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
        return output


class StandardTransformer(nn.Module):
    """标准Transformer模型实现 - 唯一的基线模型"""
    def __init__(self, input_dim=6, d_model=None, n_heads=None, e_layers=None, d_layers=None,
                 d_ff=None, dropout=None):
        super().__init__()
        # 动态导入配置以避免循环导入
        from . import HGVConfig
        config = HGVConfig.get_model_config('transformer')
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
        
        # 标准Transformer编码器（与 PLGAFormer 一致：activation='gelu', norm_first=True）
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.n_heads, self.d_ff, self.dropout, batch_first=True,
            activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.e_layers)
        
        # 标准Transformer解码器（与 PLGAFormer 一致：activation='gelu', norm_first=True）
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_model, self.n_heads, self.d_ff, self.dropout, batch_first=True,
            activation='gelu', norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, self.d_layers)
        
        # 输出投影
        self.projection = nn.Linear(self.d_model, 3)
        
    def forward(self, src, tgt=None, tgt_mask=None, target_length=None):
        if tgt is None:
            if target_length is None:
                raise ValueError("target_length is required when tgt is None")
            last_state = src[:, -1, :3]
            tgt = last_state.unsqueeze(1).repeat(1, target_length, 1)
        # 编码器
        enc_out = self.enc_embedding(src)
        enc_out = self.pos_encoding(enc_out)
        enc_out = self.encoder(enc_out)
        
        # 解码器
        dec_out = self.dec_embedding(tgt)
        dec_out = self.pos_encoding(dec_out)
        dec_out = self.decoder(dec_out, enc_out, tgt_mask=tgt_mask)
        
        # 输出投影
        output = self.projection(dec_out)
        
        return output


def create_baseline_model(model_type, input_dim=6, device=torch.device('cpu'), **kwargs):
    """
    创建基线模型的工厂函数
    
    Args:
        model_type: 模型类型 ('transformer' 或 'kalman')
        input_dim: 输入维度（从HGVConfig获取，默认6）
        device: 设备
        **kwargs: 其他参数
    
    Returns:
        基线模型实例
    """
    base_config = {}
    if model_type == 'transformer':
        from . import HGVConfig
        base_config = HGVConfig.get_model_config(model_type)
        base_config['input_dim'] = input_dim
    
    # 移除不兼容的参数，StandardTransformer使用特定的参数名称
    if model_type == 'transformer':
        # 转换参数名称以匹配StandardTransformer构造函数
        if 'nhead' in base_config:
            base_config['n_heads'] = base_config.pop('nhead')
        if 'num_encoder_layers' in base_config:
            base_config['e_layers'] = base_config.pop('num_encoder_layers')
        if 'num_decoder_layers' in base_config:
            base_config['d_layers'] = base_config.pop('num_decoder_layers')
        if 'dim_feedforward' in base_config:
            base_config['d_ff'] = base_config.pop('dim_feedforward')
        
        # 移除StandardTransformer不需要的参数（含 output_dim，输出固定为 3 [r,λ,φ]）
        unwanted_params = ['output_dim', 'max_seq_length', 'seq_len', 'sequence_length', 'max_len', 
                          'patch_len', 'stride', 'seg_len', 'factor', 'moving_avg']
        for param in unwanted_params:
            if param in base_config:
                base_config.pop(param)
    
    if model_type == 'transformer':
        model = StandardTransformer(**base_config)
    elif model_type == 'kalman':
        model = KalmanFilter(input_dim=input_dim, output_dim=3, **kwargs)
    elif model_type == 'kinematic':
        model = SphericalKinematicBaseline(input_dim=input_dim, output_dim=3, **kwargs)
    elif model_type == 'rotating_3dof':
        model = RotatingEarth3DOFBaseline(input_dim=input_dim, output_dim=3, **kwargs)
    elif model_type == 'dlinear':
        model = DLinear(input_dim=input_dim, output_dim=3, **kwargs)
    else:
        raise ValueError(f"Unsupported baseline model type: {model_type}.")
    
    return model.to(device)
