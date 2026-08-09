import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import os
import sys
from pathlib import Path
# 设置matplotlib字体，避免中文字体问题
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import joblib
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.data_paths import get_processed_data_dir, get_raw_trajectories_npz_path
from data_generation.raw_trajectory_io import load_raw_trajectories_as_dicts, save_raw_trajectories

GREEK_LAMBDA = "\u03bb"
GREEK_PHI = "\u03c6"
GREEK_GAMMA = "\u03b3"
GREEK_PSI = "\u03c8"

class DCBNN_HGV_Simulator:
    """
    基于DCBNN论文精确参数的HGV轨迹仿真器
    实现三种机动模型：
    1. 纵向机动模型 (Longitudinal-only maneuver)
    2. 纵向+横向转弯机动模型 (Longitudinal with lateral turning maneuver)  
    3. 纵向+横向编织机动模型 (Longitudinal with lateral weaving maneuver)
    
    === PLGAFormer 核心创新点分析 ===
    
    🎯 创新点1: 物理感知稀疏注意力 (Physics-aware Sparse Attention)
    - 相比PIT标准self-attention: 引入HGV轨迹物理特性
    - 相比DCBNN CNN架构: 保持全局建模能力但降低复杂度
    - 技术优势: O(n²) → O(n log n), 自适应窗口, 物理感知权重
    
    🎯 创新点2: 门控自适应物理校正器 (Gated Adaptive Physics Correction)
    - 相比PIT软约束: 硬约束直接校正物理违规
    - 相比DCBNN固定双通道: 自适应校正强度
    - 技术优势: 实时检测, 选择性校正, 可学习参数
    
    🎯 创新点3: 多头轨迹解码器 (Multi-Head Trajectory Decoder)
    - 相比PIT单输出头: 针对不同物理量专门化
    - 相比DCBNN: 头间注意力交互 + 物理约束后处理
    - 技术优势: 变量特异性建模, 自适应权重, 物理一致性
    """
    
    def __init__(self):
        # 物理常数 - 根据对比表格修正参数
        self.m = 907.0                   # 质量 (kg) - DCBNN表1
        self.S = 0.4839                  # 参考面积 (m²) - DCBNN表1
        self.g0 = 9.81                   # 标准重力加速度
        self.R_earth = 6378000           # 地球半径 (m) - 统一使用6378km标准值
        self.rho0 = 1.225                # 海平面大气密度 (kg/m³)
        self.H_scale = 7200              # 大气标高 (m)
        
        # HGV参数 - DCBNN论文表1精确值
        # self.m = 907.0                   # 质量 (kg) - DCBNN表1
        # self.S = 0.4839                  # 参考面积 (m²) - DCBNN表1
        # 移除参考长度设置，DCBNN论文未明确提及
        
        # 气动系数模型 - 根据对比表格：DCBNN使用非线性函数，PIT使用1.85
        # CL = f1(Ma, α), CD = f2(Ma, α)
        self.CL_coeffs = [0.15, 1.0, 0.12]  # [基础值, 线性项, 二次项] - 调整系数
        self.CD_coeffs = [0.04, 1.2, 0.25]  # [基础值, 二次项系数, 马赫数因子] - 调整系数
        
        # 控制参数 - DCBNN论文图2精确范围
        self.alpha_max = np.deg2rad(25.0)  # 最大攻角 25° (DCBNN图2)
        self.alpha_min = np.deg2rad(11.0)  # 最小攻角 11° (DCBNN图2)
        self.bank_max = np.deg2rad(30.0)   # 最大滚转角 30° (DCBNN论文)
        
        # 机动参数 - 参考PIT论文图2
        self.turning_duration = 200.0      # turning机动持续时间 (s)
        self.spiral_duration = 300.0       # spiral机动持续时间 (s)
        self.longitudinal_duration = 150.0 # longitudinal机动持续时间 (s)
        
        # 速度阈值 - DCBNN图1中的V1, V2
        self.V1 = 3000  # 下速度阈值 (m/s)
        self.V2 = 5000  # 上速度阈值 (m/s)
        
        # 机动类型参数
        self.maneuver_types = ['longitudinal', 'turning', 'weaving']
        self.sampling_interval_s = 1.0
        self.points_per_trajectory = 1000
        
        # ===== PIT论文环境参数设置 (Parameter Settings of Radar and Environment) =====
        # 雷达参数
        self.radar_freq = 10e9              # 雷达频率 10 GHz (PIT论文)
        self.radar_power = 1e6              # 雷达功率 1 MW (PIT论文)
        self.radar_gain = 40                # 雷达天线增益 40 dB (PIT论文)
        
        # 大气环境参数 - PIT论文标准
        self.temperature_sea_level = 288.15  # 海平面温度 K (PIT论文)
        self.pressure_sea_level = 101325     # 海平面压力 Pa (PIT论文)
        self.air_gas_constant = 287          # 空气气体常数 J/(kg·K) (PIT论文)
        self.specific_heat_ratio = 1.4       # 比热比 (PIT论文)
        
        # 地球模型参数 - PIT论文标准
        self.earth_rotation_rate = 7.292115e-5  # 地球自转角速度 rad/s (PIT论文)
        self.gravitational_parameter = 3.986004418e14  # 地球引力参数 m³/s² (PIT论文)
        
        # 机动类型参数
        
    def standard_gravity_model(self, h):
        """标准重力模型 - 使用PIT论文的引力参数"""
        return self.gravitational_parameter / (self.R_earth + h)**2
        
    def atmospheric_model(self, h):
        """Return density and sound speed from the 1976 standard atmosphere.

        The geopotential-altitude layer model is used through 84.852 km. Above
        that altitude, pressure is continued exponentially at the terminal
        layer temperature. The formal HGV envelope remains below 80 km.
        """
        altitude = max(float(h), 0.0)
        base_altitudes = np.asarray(
            [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0]
        )
        base_temperatures = np.asarray(
            [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
        )
        base_pressures = np.asarray(
            [101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420, 0.3734]
        )
        lapse_rates = np.asarray([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020])
        gas_constant = 287.05287
        gravity = 9.80665

        if altitude >= base_altitudes[-1]:
            temperature = float(base_temperatures[-1])
            pressure = float(base_pressures[-1]) * np.exp(
                -gravity * (altitude - base_altitudes[-1]) / (gas_constant * temperature)
            )
        else:
            layer = int(np.searchsorted(base_altitudes, altitude, side="right") - 1)
            layer = min(max(layer, 0), len(lapse_rates) - 1)
            h0 = float(base_altitudes[layer])
            t0 = float(base_temperatures[layer])
            p0 = float(base_pressures[layer])
            lapse = float(lapse_rates[layer])
            if abs(lapse) < 1e-12:
                temperature = t0
                pressure = p0 * np.exp(-gravity * (altitude - h0) / (gas_constant * t0))
            else:
                temperature = t0 + lapse * (altitude - h0)
                pressure = p0 * (t0 / temperature) ** (gravity / (gas_constant * lapse))

        density = pressure / (gas_constant * temperature)
        sound_speed = np.sqrt(self.specific_heat_ratio * gas_constant * temperature)
        return float(density), float(sound_speed)

    def aerodynamic_coefficients(self, alpha, bank, Ma):
        """Return the published HGV lift/drag polynomial fit used by this project.

        The fit uses angle of attack in degrees and Mach number. Bank angle does
        not alter the aerodynamic coefficients; it rotates the lift vector in
        the 3-DOF equations. The third return value is retained as a zero side-
        force coefficient for backward-compatible diagnostics.
        """
        del bank
        alpha_deg = float(np.rad2deg(alpha))
        mach = float(Ma)
        alpha_eval = float(np.clip(alpha_deg, 0.0, 25.0))
        mach_eval = float(np.clip(mach, 5.0, 25.0))

        cl = (
            -0.0561
            - 0.00443 * mach_eval
            + 0.05 * alpha_eval
            - 0.00083 * mach_eval * alpha_eval
            + 0.00032 * mach_eval**2
            + 0.00037 * alpha_eval**2
        )
        cd = (
            0.12721
            - 0.01542 * mach_eval
            + 0.00486 * alpha_eval
            - 0.00030 * mach_eval * alpha_eval
            + 0.00057 * mach_eval**2
            + 0.00067 * alpha_eval**2
        )
        return float(cl), float(max(cd, 1e-4)), 0.0
    
    def control_inputs(self, t, maneuver_type, V, h):
        """
        控制输入策略 - 严格按照DCBNN Figure 2的清晰控制逻辑
        回退到简洁有效的控制策略
        """
        # ===== 攻角控制 - DCBNN Figure 2的精确三段式控制 =====
        
        # DCBNN论文的精确速度阈值和攻角参数
        V1 = 3000.0  # 下速度阈值 (m/s)
        V2 = 5000.0  # 上速度阈值 (m/s)  
        alpha_max = 23.0  # 最大攻角 (度) - 优化差异化
        alpha_LD = 9.0    # 最小攻角 9° - 优化差异化 
        
        # 时间分段控制 - DCBNN Figure 2的精确模式
        if t <= 450.0:
            # 第一阶段：初始巡航阶段 (0-450秒)
            alpha_deg = alpha_max
            
        elif t <= 1250.0:
            # 第二阶段：中间滑翔阶段 (450-1250秒) 
            # 严格单调下降 + 适度波动，绝不允许上升
            time_ratio = (t - 450.0) / (1250.0 - 450.0)
            base_alpha = alpha_max - (alpha_max - alpha_LD) * time_ratio
            
            # DCBNN风格的波动 - 严格下降趋势
            wave_period = 200.0
            wave_phase = ((t - 450.0) % wave_period) / wave_period * 2 * np.pi
            
            # 确保波动不会让攻角超过基础下降趋势
            max_wave_amplitude = min(1.0, (base_alpha - alpha_LD) * 0.3)  # 波动幅度受限
            wave_amplitude = max_wave_amplitude * (1.0 - time_ratio * 0.8)
            
            # 只允许向下的波动
            wave_component = -abs(wave_amplitude * np.sin(wave_phase)) * 0.5
            
            alpha_deg = base_alpha + wave_component
            
            # 严格确保不超过当前时刻的基础值
            alpha_deg = min(alpha_deg, base_alpha)
            alpha_deg = np.clip(alpha_deg, alpha_LD, alpha_max)
            
        else:
            # 第三阶段：最终滑翔阶段 (1250-2000秒)
            alpha_deg = alpha_LD
        
        # 攻角物理约束
        alpha_deg = np.clip(alpha_deg, 8.0, 28.0)
        alpha = np.deg2rad(alpha_deg)
        
        # ===== 滚转角控制 - DCBNN Figure 2的清晰机动模式 =====
        
        if maneuver_type == 'longitudinal':
            # 纵向机动：强制滚转角为0°，抑制横向运动
            bank = 0.0
            
        elif maneuver_type == 'turning':
            # 转弯机动：严格按照DCBNN Figure 2(d)的恒定Bank角
            if 50.0 <= t <= 1050.0:
                bank_deg = 18.0  # 恒定18°，严格按照DCBNN论文
                bank = np.deg2rad(bank_deg)
            else:
                bank = 0.0
                
        elif maneuver_type == 'weaving':
            # 编织机动：基于论文公式(14)的weaving机动模型
            # 论文公式：v = {(-1)^k * v2, t ∈ [300 + 500k, 500 + 500k]
            #                v0,           others
            #                (-1)^(k+1) * v2, t ∈ [550 + 500k, 750 + 500k]}
            # 其中k为控制周期数，v2为weaving baseline（30°）
            
            if t >= 50.0:  # 从50秒开始机动（缩短直线距离）
                v0 = 0.0  # weaving baseline为0（"others"情况）
                v2_base = 28.0  # 基础weaving角度
                bank_deg = v0  # 默认为v0
                
                # 计算当前处于第几个控制周期 (k)
                k = int((t - 50.0) // 500.0)
                
                # 优化：随着周期数增加，逐渐减小bank角幅度，避免后续S型过于急促
                # 第一个S保持原幅度，后续S型逐渐减小
                if k == 0:
                    amplitude_factor = 1.0  # 第一个S型保持原幅度
                elif k == 1:
                    amplitude_factor = 0.75  # 第二个S型减小到75%
                elif k == 2:
                    amplitude_factor = 0.55  # 第三个S型减小到55%
                else:
                    amplitude_factor = 0.4   # 后续S型进一步减小
                
                v2 = v2_base * amplitude_factor
                
                # 计算在当前周期内的时间
                cycle_time = (t - 50.0) % 500.0
                
                # 第一机动段：[300 + 500k, 500 + 500k] 对应 [0, 200]
                if 0.0 <= cycle_time < 200.0:
                    bank_deg = ((-1) ** k) * v2
                
                # 第一恢复段：[500 + 500k, 550 + 500k] 对应 [200, 250]
                elif 200.0 <= cycle_time < 250.0:
                    bank_deg = v0  # "others" - 恢复阶段
                
                # 第二机动段：[550 + 500k, 750 + 500k] 对应 [250, 450]
                elif 250.0 <= cycle_time < 450.0:
                    bank_deg = ((-1) ** (k + 1)) * v2
                
                # 第二恢复段：[750 + 500k, 800 + 500k] 对应 [450, 500]
                else:  # 450.0 <= cycle_time < 500.0
                    bank_deg = v0  # "others" - 恢复阶段
                
                bank = np.deg2rad(bank_deg)
            else:
                bank = 0.0
        else:
            bank = 0.0
        
        # 滚转角物理约束
        bank = np.clip(bank, np.deg2rad(-35.0), np.deg2rad(35.0))
        
        return alpha, bank
    

    def hgv_dynamics(self, t, state, maneuver_type):
        """Rotating-spherical-Earth 3-DOF point-mass equations.

        State order is ``[r, longitude, latitude, V, gamma, psi]`` in SI
        units. ``V`` is Earth-relative speed, ``gamma`` is flight-path angle,
        and ``psi`` is heading measured clockwise from north. No derivative
        clipping or maneuver-specific damping is applied in this formal path.
        """
        state_arr = np.asarray(state, dtype=float)
        if state_arr.shape != (6,) or not np.all(np.isfinite(state_arr)):
            raise ValueError("HGV state must contain six finite values.")

        r, longitude, latitude, velocity, gamma, psi = state_arr
        del longitude
        if r <= self.R_earth or velocity <= 0.0:
            raise ValueError("HGV dynamics require positive altitude and speed.")

        altitude = r - self.R_earth
        density, sound_speed = self.atmospheric_model(altitude)
        mach = velocity / sound_speed
        alpha, bank = self.control_inputs(t, maneuver_type, velocity, altitude)
        cl, cd, _ = self.aerodynamic_coefficients(alpha, bank, mach)

        dynamic_pressure = 0.5 * density * velocity**2
        lift = cl * dynamic_pressure * self.S
        drag = cd * dynamic_pressure * self.S
        gravity = self.standard_gravity_model(altitude)
        omega = self.earth_rotation_rate

        sin_gamma = np.sin(gamma)
        cos_gamma = np.cos(gamma)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        sin_lat = np.sin(latitude)
        cos_lat = np.cos(latitude)
        safe_cos_gamma = np.copysign(max(abs(cos_gamma), 1e-8), cos_gamma)
        safe_cos_lat = np.copysign(max(abs(cos_lat), 1e-8), cos_lat)

        dr_dt = velocity * sin_gamma
        dlongitude_dt = velocity * cos_gamma * sin_psi / (r * safe_cos_lat)
        dlatitude_dt = velocity * cos_gamma * cos_psi / r

        dvelocity_dt = (
            -drag / self.m
            - gravity * sin_gamma
            + omega**2
            * r
            * cos_lat
            * (sin_gamma * cos_lat - cos_gamma * sin_lat * cos_psi)
        )
        dgamma_dt = (
            lift * np.cos(bank) / (self.m * velocity)
            - (gravity / velocity - velocity / r) * cos_gamma
            + 2.0 * omega * cos_lat * sin_psi
            + omega**2
            * r
            * cos_lat
            / velocity
            * (cos_gamma * cos_lat + sin_gamma * sin_lat * cos_psi)
        )
        dpsi_dt = (
            lift * np.sin(bank) / (self.m * velocity * safe_cos_gamma)
            + velocity * cos_gamma * sin_psi * np.tan(latitude) / r
            + 2.0 * omega * sin_lat
            + omega**2
            * r
            * sin_psi
            * sin_lat
            * cos_lat
            / (velocity * safe_cos_gamma)
            - 2.0 * omega * cos_lat * cos_psi * np.tan(gamma)
        )

        derivatives = np.asarray(
            [dr_dt, dlongitude_dt, dlatitude_dt, dvelocity_dt, dgamma_dt, dpsi_dt],
            dtype=float,
        )
        if not np.all(np.isfinite(derivatives)):
            raise FloatingPointError("Non-finite derivative in formal HGV dynamics.")
        return derivatives.tolist()
    
    def validate_trajectory(self, trajectory, maneuver_type='turning'):
        """
        球坐标轨迹物理合理性验证
        """
        if trajectory is None or len(trajectory) == 0:
            print("❌ 轨迹为空")
            return False
            
        # 检查NaN或无限值
        if np.any(~np.isfinite(trajectory)):
            print("❌ 轨迹包含NaN或无限值")
            return False
            
        # 提取关键物理量 - 球坐标 [r, λ, φ, V, γ, ψ]
        r = trajectory[:, 0]        # 地心距离
        lambda_coords = trajectory[:, 1] # 经度
        phi_coords = trajectory[:, 2]  # 纬度
        V = trajectory[:, 3]        # 速度
        gamma = trajectory[:, 4]    # 航迹角
        psi = trajectory[:, 5]      # 航向角
        
        # 使用类属性中的地球半径（确保一致性）
        R_earth = self.R_earth
        # 计算高度用于验证
        h = r - R_earth
        
        # 1. 地心距离约束检查
        if np.any(r < R_earth):
            print(f"❌ 轨迹包含小于地球半径的地心距离: 最小值 {np.min(r):.2f}m < {R_earth}m")
            return False
        if np.any(h > 120000):
            print(f"❌ 轨迹高度超出范围: 最大值 {np.max(h):.2f}m > 120km")
            return False
            
        # 2. 速度约束检查 - 合理的HGV速度范围
        if np.any(V < 100):
            print(f"❌ 轨迹速度过低: 最小值 {np.min(V):.2f}m/s < 100m/s")
            return False
        if np.any(V > 15000):
            print(f"❌ 轨迹速度过高: 最大值 {np.max(V):.2f}m/s > 15000m/s")
            return False
            
        # 3. 角度约束检查
        if np.any(np.abs(gamma) > np.pi/2):
            print(f"❌ 航迹角超出范围: 最大值 {np.max(np.abs(gamma)):.3f} > π/2")
            return False
        if np.any(np.abs(lambda_coords) > 2*np.pi):
            print(f"❌ 经度超出范围: 最大值 {np.max(np.abs(lambda_coords)):.3f} > 2π")
            return False
        if np.any(np.abs(phi_coords) > np.pi/2):
            print(f"❌ 纬度超出范围: 最大值 {np.max(np.abs(phi_coords)):.3f} > π/2")
            return False
            
        # 4. 位置变化合理性检查 - 球坐标系
        if len(trajectory) > 1:
            # 使用类属性中的地球半径（确保一致性）
            R_earth = self.R_earth
            
            # 计算地面距离变化（使用球面距离公式）
            lat1, lon1 = phi_coords[0], lambda_coords[0]
            lat2, lon2 = phi_coords[-1], lambda_coords[-1]
            
            # Haversine公式计算球面距离
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            ground_distance = R_earth * c
            
            # 高度变化
            height_change = abs(h[-1] - h[0])
            
            # 总位移（3D距离）
            total_displacement = np.sqrt(ground_distance**2 + height_change**2)
            
            # 检查是否有明显的位移 - 根据机动类型调整
            min_displacement = 10000  # 默认10km
            if maneuver_type == 'longitudinal':
                min_displacement = 5000  # 纵向机动允许更小的总位移
            elif maneuver_type == 'weaving':
                min_displacement = 8000  # 编织机动中等位移要求
            
            if total_displacement < min_displacement:
                print(f"❌ 轨迹总位移过小: {total_displacement:.2f}m < {min_displacement/1000:.1f}km ({maneuver_type})")
                return False
                
            # 检查是否位移过大
            if total_displacement > 50000000:  # 大于50000km总位移  
                print(f"❌ 轨迹总位移过大: {total_displacement/1000:.2f}km > 50000km")
                return False
        
        # 5. 加速度物理合理性检查
        if len(trajectory) > 1:
            dt = float(getattr(self, 'sampling_interval_s', 1.0))
            
            # 速度变化率
            dv = np.diff(V)
            accelerations = dv / dt
            max_accel = np.max(np.abs(accelerations))
            
            if max_accel > 200:  # 放宽加速度限制
                print(f"❌ 轨迹加速度异常: 最大加速度 {max_accel:.2f}m/s² > 200m/s²")
                return False
                
        # 6. 高度变化合理性检查 - 放宽限制适应长时间仿真
        if len(trajectory) > 1:
            height_change = np.abs(h[-1] - h[0])
            if height_change > 60000:  # 高度变化不应超过60km（放宽）
                print(f"❌ 高度变化异常: {height_change/1000:.1f}km > 60km")
                return False
        
        # 7. 马赫数检查 - 放宽范围
        if len(trajectory) > 1:
            # 估算马赫数
            sound_speeds = []
            for altitude in h[::10]:  # 每10个点检查一次以提高效率
                rho, a = self.atmospheric_model(altitude)
                sound_speeds.append(a)
            
            mach_numbers = V[::10] / np.array(sound_speeds)
            
            # HGV典型马赫数范围：Ma 1-30 (放宽)
            if np.any(mach_numbers < 1.0) or np.any(mach_numbers > 30):
                min_ma, max_ma = np.min(mach_numbers), np.max(mach_numbers)
                print(f"❌ 马赫数超出合理范围: [{min_ma:.1f}, {max_ma:.1f}] vs [1.0, 30]")
                return False
        
        # 所有检查通过
        return True
    
    def simulate_trajectory(self, initial_conditions, maneuver_type, duration=999.0, sampling_interval_s=None, num_points=None):
        """
        高精度轨迹仿真 - 使用球坐标系统
        """
        # 高精度轨迹仿真
        
        # 初始条件设置 - 球坐标系状态变量 [r, λ, φ, V, γ, ψ]
        # 使用类属性中的地球半径（确保一致性）
        R_earth = self.R_earth
        
        # 兼容旧的h参数，自动转换为r
        if 'h' in initial_conditions and 'r' not in initial_conditions:
            r0 = initial_conditions['h'] + R_earth  # 高度转换为地心距离
        else:
            r0 = initial_conditions.get('r', R_earth + 65000)  # 初始地心距离 (m)
            
        lon0 = initial_conditions.get('lambda', initial_conditions.get(GREEK_LAMBDA, 0.0))    # 初始经度 (rad)
        lat0 = initial_conditions.get('phi', initial_conditions.get(GREEK_PHI, 0.0))       # 初始纬度 (rad)
        V0 = initial_conditions.get('V', 6500)          # 初始速度 (m/s)
        gamma0 = initial_conditions.get('gamma', initial_conditions.get(GREEK_GAMMA, -0.02)) # 初始航迹角 (rad)
        psi0 = initial_conditions.get('psi', initial_conditions.get(GREEK_PSI, 0.0))       # 初始航向角 (rad)
        
        y0_state = [r0, lon0, lat0, V0, gamma0, psi0]
        
        # 时间序列
        sampling_interval_s = float(
            sampling_interval_s if sampling_interval_s is not None else getattr(self, 'sampling_interval_s', 1.0)
        )
        if sampling_interval_s <= 0:
            raise ValueError("sampling_interval_s must be positive.")
        if num_points is None:
            num_points = int(round(float(duration) / sampling_interval_s)) + 1
        num_points = int(num_points)
        if num_points < 2:
            raise ValueError("num_points must be at least 2.")
        t_eval = np.arange(num_points, dtype=float) * sampling_interval_s
        duration = float(t_eval[-1])
        t_span = (0.0, duration)
            
        try:
            # 输入参数验证
            if not all(np.isfinite([r0, lon0, lat0, V0, gamma0, psi0])):
                print(f"❌ 初始条件包含无效值: r={r0}, λ={lon0}, φ={lat0}, V={V0}, γ={gamma0}, ψ={psi0}")
                return None, None
                
            if V0 <= 0 or r0 <= R_earth:
                print(f"❌ 初始条件不合理: V={V0}, r={r0}")
                return None, None
            
            # 高精度数值积分
            sol = solve_ivp(lambda t, y: self.hgv_dynamics(t, y, maneuver_type),
                           t_span, y0_state, t_eval=t_eval,
                           method='RK45',
                           rtol=1e-5, atol=1e-7,    # 适中的精度，避免过度严格
                           max_step=max(0.1, sampling_interval_s),
                           first_step=min(0.2, sampling_interval_s))          # 增大初始步长
            
            if not sol.success:
                print(f"❌ 数值积分失败 ({maneuver_type}): {sol.message}")
                return None, None
            
            t = sol.t
            trajectory = sol.y.T
            
            # 检查轨迹数据完整性
            if len(t) == 0 or len(trajectory) == 0:
                print(f"❌ 轨迹数据为空 ({maneuver_type})")
                return None, None
                
            if np.any(~np.isfinite(trajectory)):
                print(f"❌ 轨迹包含无效数值 ({maneuver_type})")
                return None, None
            
            # 使用完整的轨迹验证函数
            if self.validate_trajectory(trajectory, maneuver_type):
                return t, trajectory
            else:
                print(f"❌ 轨迹验证失败 ({maneuver_type})")
                return None, None
                
        except ValueError as e:
            print(f"❌ 数值错误 ({maneuver_type}): {str(e)}")
            return None, None
        except RuntimeError as e:
            print(f"❌ 运行时错误 ({maneuver_type}): {str(e)}")
            return None, None
        except Exception as e:
            print(f"❌ 未知错误 ({maneuver_type}): {type(e).__name__}: {str(e)}")
            return None, None

    def generate_trajectories(self, count=1000):
        """
        Generate complete HGV trajectories before split/window construction.
        """
        print(f"🚀 Generating {count} trajectories...")
        
        trajectories = []
        maneuver_types = ['longitudinal', 'turning', 'weaving']
        
        # PIT论文标准的机动类型分布
        maneuver_distribution = {
            'longitudinal': count // 3,      # 约213条
            'turning': count // 3,           # 约213条  
            'weaving': count - 2 * (count // 3)  # 约214条
        }
        
        total_generated = 0
        for maneuver_type, num_trajectories in maneuver_distribution.items():
            print(f"\n📈 Generating {maneuver_type} maneuvers: {num_trajectories} trajectories")
            
            successful_count = 0
            attempt_count = 0
            max_attempts = num_trajectories * 2  # 允许最多2倍尝试次数
            
            while successful_count < num_trajectories and attempt_count < max_attempts:
                attempt_count += 1
                
                try:
                    # 球坐标系统的初始条件范围 - 基于标准航空符号
                    # 根据机动类型调整初始条件范围，提高成功率
                    if maneuver_type == 'longitudinal':
                        # 纵向机动：更保守的初始条件
                        initial_conditions = {
                            'h': np.random.uniform(62000, 72000),               # 高度: 62-72km
                            'λ': np.random.uniform(-0.05, 0.05),                # 经度: ±0.05 rad
                            'φ': np.random.uniform(-0.05, 0.05),                # 纬度: ±0.05 rad
                            'V': np.random.uniform(5800, 6200),                 # 速度: 5800-6200m/s
                            'γ': np.random.uniform(-0.03, -0.015),              # 航迹角: 温和下降
                            'ψ': np.random.uniform(-0.05, 0.05),                # 航向角: ±0.05 rad
                        }
                    else:
                        # 转弯和编织机动：标准初始条件
                        initial_conditions = {
                            'h': np.random.uniform(60000, 75000),               # 高度: 60-75km
                            'λ': np.random.uniform(-0.1, 0.1),                  # 经度: ±0.1 rad
                            'φ': np.random.uniform(-0.1, 0.1),                  # 纬度: ±0.1 rad
                            'V': np.random.uniform(5500, 6500),                 # 速度: 5500-6500m/s
                            'γ': np.random.uniform(-0.05, -0.01),               # 航迹角: 初始下降
                            'ψ': np.random.uniform(-0.1, 0.1),                  # 航向角: ±0.1 rad
                        }
                    
                    # 仿真时长
                    sampling_interval_s = float(getattr(self, 'sampling_interval_s', 1.0))
                    num_points = int(getattr(self, 'points_per_trajectory', 1000))
                    duration = getattr(self, 'custom_duration', (num_points - 1) * sampling_interval_s)
                    
                    # 仿真轨迹
                    t, trajectory = self.simulate_trajectory(
                        initial_conditions,
                        maneuver_type,
                        duration,
                        sampling_interval_s=sampling_interval_s,
                        num_points=num_points,
                    )
                    
                except Exception as e:
                    print(f"    ⚠️  初始条件生成错误 ({maneuver_type}): {str(e)}")
                    continue
                
                if t is not None and trajectory is not None:
                    # 球坐标系统数据格式 [r, λ, φ, V, γ, ψ]
                    traj_dict = {
                        'r': trajectory[:, 0],          # 地心距离
                        'λ': trajectory[:, 1],          # 经度
                        'φ': trajectory[:, 2],          # 纬度
                        'V': trajectory[:, 3],          # 速度
                        'γ': trajectory[:, 4],          # 航迹角
                        'ψ': trajectory[:, 5],          # 航向角
                        'time': t,
                        'velocity': trajectory[:, 3],   # 速度直接从状态变量获取
                        'alpha': [self.control_inputs(ti, maneuver_type, 
                                trajectory[i, 3], trajectory[i, 0] - self.R_earth)[0] for i, ti in enumerate(t)],
                        'bank': [self.control_inputs(ti, maneuver_type, 
                               trajectory[i, 3], trajectory[i, 0] - self.R_earth)[1] for i, ti in enumerate(t)],
                        'maneuver_type': maneuver_type
                    }
                    trajectories.append(traj_dict)
                    successful_count += 1
                    total_generated += 1
                    
                    # 进度报告（每20条轨迹报告一次，避免重复显示）
                    if successful_count % 20 == 0 and successful_count > 0:
                        progress_percent = 100 * total_generated / count
                        print(f"    {maneuver_type}: {successful_count}/{num_trajectories} | 总进度: {total_generated}/{count} ({progress_percent:.1f}%)")
                
                # 如果尝试次数过多，给出警告（避免频繁输出）
                if attempt_count % 100 == 0 and attempt_count > 0:
                    success_rate = 100 * successful_count / attempt_count if attempt_count > 0 else 0
                    print(f"    警告 - {maneuver_type}: {attempt_count} 次尝试, {successful_count} 次成功 (成功率: {success_rate:.1f}%)")
            
            print(f"{maneuver_type} 完成: {successful_count}/{num_trajectories} 条轨迹")
            
            if successful_count < num_trajectories:
                print(f"  警告: 仅生成了 {successful_count}/{num_trajectories} 条{maneuver_type}轨迹")
        
        print(f"数据集生成完成!")
        print(f"最终统计:")
        print(f"    轨迹总数: {len(trajectories)}")
        print(f"    成功率: {100*len(trajectories)/count:.1f}%")
        
        return trajectories

    
    def create_pit_dataset(self, trajectories, times, labels, seq_len=256, pred_len=256):
        """
        创建PLGAFormer数据集格式；正式协议由 seq_len/pred_len 参数控制。
        """
        # 创建PIT论文格式数据集
        
        X, y, maneuver_labels = [], [], []
        
        for traj, t, label in zip(trajectories, times, labels):
            if len(traj) < seq_len + pred_len:
                continue
            
            # 球坐标特征：[r, λ, φ, V, γ, ψ] - 6维特征
            features = traj[:, :6]
            
            # 创建滑动窗口
            for i in range(len(features) - seq_len - pred_len + 1):
                # 输入序列：seq_len时间步 × 6特征 (球坐标系统 [r, λ, φ, V, γ, ψ])
                input_seq = features[i:i+seq_len]
                
                # 输出序列：pred_len时间步 × 3特征 (预测位置 [r, λ, φ])
                output_seq = features[i+seq_len:i+seq_len+pred_len, :3]
                
                X.append(input_seq)
                y.append(output_seq)
                maneuver_labels.append(label)
        
        X = np.array(X)
        y = np.array(y)
        maneuver_labels = np.array(maneuver_labels)
        
        print(f"数据集统计信息:")
        print(f"    输入形状: {X.shape}")
        print(f"    输出形状: {y.shape}")
        print(f"    样本总数: {len(X):,}")
        print(f"    机动类型分布: {dict(zip(*np.unique(maneuver_labels, return_counts=True)))}")
        
        return X, y, maneuver_labels

    def generate_trajectory_plots(self, trajectories, times, labels):
        """
        生成轨迹可视化图 - 简化版本，重点展示轨迹和控制参数
        """
        # 生成轨迹可视化图
        
        # 创建DCBNN Figure 2样式的图形布局
        fig = plt.figure(figsize=(18, 12))
        
        maneuver_types = ['longitudinal', 'turning', 'weaving']
        colors = ['blue', 'red', 'green']
        names = {
            'longitudinal': 'Longitudinal-only maneuver',
            'turning': 'Longitudinal with lateral turning maneuver',
            'weaving': 'Longitudinal with lateral weaving maneuver'
        }
        
        subplot_idx = 1
        for maneuver_type, color in zip(maneuver_types, colors):
            # 找到该机动类型的轨迹
            traj_indices = [i for i, label in enumerate(labels) if label == maneuver_type]
            if not traj_indices:
                continue
                
            trajectory = trajectories[traj_indices[0]]
            t = times[traj_indices[0]]
            
            # 提取球坐标轨迹数据 [r, λ, φ, V, γ, ψ]
            r_traj = trajectory[:, 0] / 1000  # 地心距离 km
            lon_traj = np.rad2deg(trajectory[:, 1])  # 经度 度
            lat_traj = np.rad2deg(trajectory[:, 2])  # 纬度 度
            
            # 3D球坐标轨迹图 (左列)
            ax3d = fig.add_subplot(2, 3, subplot_idx, projection='3d')
            # 计算高度用于Z轴显示
            h_traj = (r_traj * 1000 - self.R_earth) / 1000  # 高度 km
            ax3d.plot(lon_traj, lat_traj, h_traj, color=color, linewidth=2)
            ax3d.set_xlabel('Longitude (°)')
            ax3d.set_ylabel('Latitude (°)')
            ax3d.set_zlabel('Altitude (km)')
            ax3d.set_title(f'({chr(97+(subplot_idx-1)*2)}) {names[maneuver_type]} - 3D轨迹')
            ax3d.grid(True, alpha=0.3)
            
            # 控制参数图 (右列)
            ax_ctrl = fig.add_subplot(2, 3, subplot_idx+3)
            
            # 计算控制参数
            alpha_list, bank_list = [], []
            for time_val in t:
                time_idx = np.where(t == time_val)[0][0]
                V = trajectory[time_idx, 3]  # 球坐标系中V直接是速度大小
                h = trajectory[time_idx, 0] - self.R_earth  # r转换为高度h
                alpha, bank = self.control_inputs(time_val, maneuver_type, V, h)
                alpha_list.append(np.rad2deg(alpha))
                bank_list.append(np.rad2deg(bank))
            
            ax_ctrl.plot(t, alpha_list, color='red', linewidth=2, label='Angle of Attack α (°)')
            ax_ctrl.plot(t, bank_list, color='green', linewidth=2, linestyle='--', label='Bank Angle σ (°)')
            ax_ctrl.set_xlabel('Time (s)')
            ax_ctrl.set_ylabel('Control Angle (°)')
            ax_ctrl.set_title(f'({chr(97+(subplot_idx-1)*2+1)}) 控制参数')
            ax_ctrl.grid(True, alpha=0.3)
            ax_ctrl.legend()
            ax_ctrl.set_xlim(0, max(t))
            
            subplot_idx += 1
        
        plt.tight_layout()
        
        # 保存图像
        os.makedirs('figures', exist_ok=True)
        figure_path = 'figures/hgv_trajectories_visualization.png'
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"轨迹可视化图已保存到: {figure_path}")
        
        # 生成速度-高度图
        self.generate_velocity_altitude_plot(trajectories, save_path='figures/velocity_altitude_relationship.png')
    
    def visualize_trajectories(self, trajectories, save_path='figures/hgv_trajectories_visualization.png'):
        """
        生成DCBNN论文Figure 2风格的可视化
        显示三种机动模型的3D轨迹和控制参数
        """
        print("📈 Generating DCBNN Figure 2 style visualization...")
        
        # Create figure with 2x3 subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Three maneuver types
        maneuver_types = ['longitudinal', 'turning', 'weaving']
        maneuver_names = ['Longitudinal-only', 'Longitudinal with Lateral Turning', 'Longitudinal with Lateral Weaving']
        colors = ['blue', 'red', 'green']
        
        for idx, (maneuver_type, name, color) in enumerate(zip(maneuver_types, maneuver_names, colors)):
            # Filter trajectories by maneuver type
            maneuver_trajectories = [t for t in trajectories if t['maneuver_type'] == maneuver_type]
            
            if not maneuver_trajectories:
                continue
                
            # Use the first trajectory for visualization
            traj = maneuver_trajectories[0]
            
            # Extract spherical coordinate trajectory data
            r_traj = traj['r'] / 1000  # Convert to km (geocentric distance)
            lon_traj = np.rad2deg(traj['λ'])  # Convert to degrees (longitude)
            lat_traj = np.rad2deg(traj['φ'])  # Convert to degrees (latitude)
            t = traj['time']
            alpha_list = np.rad2deg(traj['alpha'])  # Convert to degrees
            bank_list = np.rad2deg(traj['bank'])    # Convert to degrees
            
            # 3D spherical coordinate trajectory plot (left column: a, c, e)
            ax3d = fig.add_subplot(2, 3, idx+1, projection='3d')
            # Convert geocentric distance to altitude for display
            h_traj = (r_traj * 1000 - self.R_earth) / 1000  # altitude in km
            ax3d.plot(lon_traj, lat_traj, h_traj, color=color, linewidth=2, label=f'{name} trajectory')
            ax3d.set_xlabel('Longitude (°)')
            ax3d.set_ylabel('Latitude (°)')
            ax3d.set_zlabel('Altitude (km)')
            ax3d.set_title(f'({chr(97+idx*2)}) {name} Model - 3D Trajectory')
            ax3d.grid(True, alpha=0.3)
            ax3d.legend()
            
            # Control parameters plot (right column: b, d, f)
            ax_ctrl = fig.add_subplot(2, 3, idx+4)
            t_plot = t[:len(alpha_list)]
            ax_ctrl.plot(t_plot, alpha_list, color=color, linewidth=2, label='Angle of Attack (°)')
            ax_ctrl.plot(t_plot, bank_list, color='orange', linewidth=2, linestyle='--', label='Bank Angle (°)')
            ax_ctrl.set_xlabel('Time (s)')
            ax_ctrl.set_ylabel('Control Angle (°)')
            ax_ctrl.set_title(f'({chr(97+idx*2+1)}) {name} Model - Control Parameters')
            ax_ctrl.grid(True, alpha=0.3)
            ax_ctrl.legend()
        
        plt.tight_layout()
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DCBNN Figure 2风格可视化已保存到: {save_path}")
        plt.show()
    
    def generate_velocity_altitude_plot(self, trajectories, save_path='figures/velocity_altitude_relationship.png'):
        """
        生成速度-高度关系图
        """
        print("生成速度-高度关系图...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        maneuver_types = ['longitudinal', 'turning', 'weaving']
        maneuver_names = ['Constant', 'Skip', 'S-turn']
        colors = ['blue', 'red', 'green']
        
        for idx, (maneuver_type, name, color) in enumerate(zip(maneuver_types, maneuver_names, colors)):
            ax = axes[idx]
            
            # Filter trajectories by maneuver type
            maneuver_trajectories = [t for t in trajectories if t['maneuver_type'] == maneuver_type]
            
            for traj in maneuver_trajectories[:5]:  # Show first 5 trajectories
                velocity = np.array(traj['V']) / 1000  # Convert to km/s (spherical coord V)
                # Convert geocentric distance to altitude
                altitude = (np.array(traj['r']) - self.R_earth) / 1000  # Convert to km
                ax.plot(velocity, altitude, color=color, alpha=0.6, linewidth=1)
            
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Altitude (km)')
            ax.set_title(f'{name} - Velocity-Altitude Relationship')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(3, 8)
            ax.set_ylim(0, 80)
        
        plt.tight_layout()
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"速度-高度关系图已保存到: {save_path}")
        plt.show()

    def generate_control_parameters_analysis(self):
        """
        生成HGV控制参数分析图 - 使用球坐标系统
        """
        # 生成HGV控制参数分析图
        
        # 固定初始条件 - 球坐标系统，与generate_3d_trajectory_plots保持一致
        base_initial_conditions = {
            'h': 65000,                         # H₀ = 65km (DCBNN Table 1)
            'λ': 0.0,                           # 初始经度 (rad)
            'φ': 0.0,                           # 初始纬度 (rad)
            'V': 6500.4,                        # 初始速度大小 (m/s) - longitudinal基准
            'γ': -0.0077,                       # 初始航迹角 (rad)
            'ψ': 0.0,                           # 初始航向角 (rad) - longitudinal基准
        }
        
        maneuver_types = ['longitudinal', 'turning', 'weaving']
        trajectories = {}
        
        duration = float(getattr(self, 'custom_duration', (getattr(self, 'points_per_trajectory', 1000) - 1) * getattr(self, 'sampling_interval_s', 1.0)))
        
        for maneuver_type in maneuver_types:
            # 生成轨迹数据
            
            # 为不同机动类型设置精确初始条件，与generate_3d_trajectory_plots保持一致
            initial_conditions = base_initial_conditions.copy()
            if maneuver_type == 'turning':
                # C型机动（turning）- 与generate_3d_trajectory_plots中c_initial一致
                initial_conditions['V'] = 6501.1
                initial_conditions['ψ'] = 0.0185
            elif maneuver_type == 'weaving':
                # S型机动（weaving）- 与generate_3d_trajectory_plots中s_initial一致
                initial_conditions['V'] = 6500.8
                initial_conditions['ψ'] = -0.0154
            else:  # longitudinal
                # 纵向机动保持基准值
                initial_conditions['V'] = 6500.4
                initial_conditions['ψ'] = 0.0
            
            # 仿真轨迹
            t, traj = self.simulate_trajectory(initial_conditions, maneuver_type, duration)
            if t is not None and traj is not None:
                # 轨迹生成成功
                # 计算控制参数历史 - 球坐标系统
                alpha_history = []
                bank_history = [] 
                for ti in t:
                    idx = np.where(t == ti)[0][0]
                    # 球坐标状态变量: [r, λ, φ, V, γ, ψ]
                    r = traj[idx, 0]   # 地心距离
                    h = r - self.R_earth  # 计算高度用于显示
                    V = traj[idx, 3]   # 速度
                    alpha, bank = self.control_inputs(ti, maneuver_type, V, h)
                    alpha_history.append(np.rad2deg(alpha))
                    bank_history.append(np.rad2deg(bank))
                
                trajectories[maneuver_type] = {
                    'time': t,
                    'trajectory': traj,
                    'alpha': alpha_history,
                    'bank': bank_history
                }
            else:
                # 轨迹生成失败
                pass
        
        print(f"📈 成功生成 {len(trajectories)} 种机动轨迹")
        for maneuver_type in trajectories:
            print(f"  - {maneuver_type}: {len(trajectories[maneuver_type]['time'])} 个数据点")
        
        if trajectories:
            # 创建Figure 2样式的6子图布局 (3行2列)
            fig = plt.figure(figsize=(12, 18))
            colors = {'longitudinal': 'blue', 'turning': 'red', 'weaving': 'green'}
            names = {
                'longitudinal': 'Longitudinal-only maneuver',
                'turning': 'Longitudinal with lateral turning maneuver', 
                'weaving': 'Longitudinal with lateral weaving maneuver'
            }
            
            subplot_index = 0
            for maneuver_type in maneuver_types:
                if maneuver_type in trajectories:
                    data = trajectories[maneuver_type]
                    t = data['time']
                    traj = data['trajectory']
                    
                    # 3D轨迹图 (左列: a, c, e) - 使用球坐标
                    ax3d = fig.add_subplot(3, 2, subplot_index * 2 + 1, projection='3d')
                    
                    # 使用球坐标系变量
                    # 球坐标状态变量: [r, λ, φ, V, γ, ψ]
                    r_coords = traj[:, 0] / 1000           # 地心距离 (km)
                    lambda_coords = traj[:, 1]             # 经度 (rad)
                    phi_coords = traj[:, 2]                # 纬度 (rad)
                    h_coords = (traj[:, 0] - self.R_earth) / 1000  # 高度 (km)
                    
                    # 直接使用球坐标进行可视化
                    # 转换角度为度数以便显示
                    lambda_deg = lambda_coords * 180 / np.pi  # 经度 (度)
                    phi_deg = phi_coords * 180 / np.pi        # 纬度 (度)
                    
                    # Nature期刊风格的机动轨迹线系统
                    z_base = np.min(h_coords) - 3  # 设置底面高度（与坐标范围一致）
                    
                    # 计算速度用于颜色映射
                    V_coords = traj[:, 3]  # 速度大小
                    V_normalized = (V_coords - np.min(V_coords)) / (np.max(V_coords) - np.min(V_coords))
                    
                    # 使用速度颜色映射的3D轨迹线
                    colors = plt.cm.plasma(V_normalized)  # 使用plasma颜色映射
                    for i in range(len(lambda_deg)-1):
                        ax3d.plot(lambda_deg[i:i+2], phi_deg[i:i+2], h_coords[i:i+2],
                                 color=colors[i], linewidth=2.5, alpha=0.95)
                    
                    # 添加颜色条
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                                              norm=plt.Normalize(vmin=np.min(V_coords), vmax=np.max(V_coords)))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax3d, shrink=0.6, aspect=20, pad=0.1)
                    cbar.set_label('Velocity (m/s)', fontsize=8)
                    
                    # 地面轨迹投影（Nature浅灰色，更透明）
                    ax3d.plot(lambda_deg, phi_deg, 
                             np.full_like(h_coords, z_base),  # 固定在底面
                             color='#d3d3d3', linewidth=1.2, linestyle='-', alpha=0.4, 
                             label='Ground Track')
                    
                    # 投影线已移除
                    
                    # 起点和终点标记（Nature期刊风格）
                    ax3d.scatter([lambda_deg[0]], [phi_deg[0]], [h_coords[0]], 
                               color='#2ca02c', s=60, marker='o', label='Start', 
                               edgecolors='#1a6b1a', linewidth=1.5, alpha=0.9)
                    ax3d.scatter([lambda_deg[-1]], [phi_deg[-1]], [h_coords[-1]], 
                               color='#d62728', s=60, marker='s', label='End', 
                               edgecolors='#a01e20', linewidth=1.5, alpha=0.9)
                    
                    # 起点到底面的垂直连线（更透明）
                    ax3d.plot([lambda_deg[0], lambda_deg[0]], [phi_deg[0], phi_deg[0]], 
                             [h_coords[0], z_base], color='#2ca02c', linewidth=1.0, alpha=0.5)
                    
                    # 终点到底面的垂直连线（更透明）
                    ax3d.plot([lambda_deg[-1], lambda_deg[-1]], [phi_deg[-1], phi_deg[-1]], 
                             [h_coords[-1], z_base], color='#d62728', linewidth=1.0, alpha=0.5)
                    
                    ax3d.set_xlabel('Lambda (°)')
                    ax3d.set_ylabel('Phi (°)')
                    ax3d.set_zlabel('Altitude (km)')
                    # 标题已移除
                    ax3d.grid(True, alpha=0.15)
                    
                    # 创建自定义图例（不包含速度颜色映射的线段）
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='#d3d3d3', linewidth=1.2, alpha=0.4, label='Ground Track'),
                        Line2D([0], [0], marker='o', color='#2ca02c', markersize=6, linestyle='None', label='Start'),
                        Line2D([0], [0], marker='s', color='#d62728', markersize=6, linestyle='None', label='End')
                    ]
                    ax3d.legend(handles=legend_elements, fontsize=7, loc='upper right')
                    
                    # 设置透明背景
                    ax3d.xaxis.pane.fill = False
                    ax3d.yaxis.pane.fill = False
                    ax3d.zaxis.pane.fill = False
                    ax3d.xaxis.pane.set_edgecolor('w')
                    ax3d.yaxis.pane.set_edgecolor('w')
                    ax3d.zaxis.pane.set_edgecolor('w')
                    ax3d.xaxis.pane.set_alpha(0.1)
                    ax3d.yaxis.pane.set_alpha(0.1)
                    ax3d.zaxis.pane.set_alpha(0.1)
                    
                    # 设置DCBNN风格的视角
                    ax3d.view_init(elev=25, azim=45)  # 稍微调整仰角
                    
                    # 使用与generate_3d_trajectory_plots相同的简单直接坐标范围设置方法
                    # 设置合理的坐标轴范围 - 参考成功的generate_3d_trajectory_plots方法
                    if maneuver_type == 'weaving':
                        # S型机动需要更大的经度轴范围
                        ax3d.set_xlim(np.min(lambda_deg) - 2, np.max(lambda_deg) + 3)
                    else:
                        # C型和纵向机动使用标准范围
                        ax3d.set_xlim(np.min(lambda_deg) - 2, np.max(lambda_deg) + 2)
                    
                    ax3d.set_ylim(np.min(phi_deg) - 1, np.max(phi_deg) + 1)
                    ax3d.set_zlim(z_base, np.max(h_coords) + 2)
                    
                    # 控制参数图 (右列: b, d, f)
                    ax_ctrl = fig.add_subplot(3, 2, subplot_index * 2 + 2)
                    
                    # 增强的控制参数可视化
                    # 攻角线条（渐变色）
                    alpha_line = ax_ctrl.plot(t, data['alpha'], color='#e74c3c', linewidth=2.5, 
                                             label='Attack Angle (α)', alpha=0.9)
                    # 滚转角线条（虚线）
                    bank_line = ax_ctrl.plot(t, data['bank'], color='#27ae60', linewidth=2.5, 
                                            linestyle='--', label='Bank Angle (σ)', alpha=0.9)
                    
                    # 添加零线参考
                    ax_ctrl.axhline(y=0, color='#34495e', linewidth=0.8, linestyle='-', alpha=0.5)
                    
                    # 计算并显示升阻比（简化版）
                    traj = data['trajectory']
                    V_vals = traj[:, 3]  # 速度
                    h_vals = traj[:, 0]  # 高度
                    
                    # 计算马赫数和升阻比
                    Ma_vals = []
                    LD_vals = []
                    for i, (v, h) in enumerate(zip(V_vals, h_vals)):
                        rho, a = self.atmospheric_model(h)
                        Ma = v / a
                        Ma_vals.append(Ma)
                        
                        # 简化的升阻比计算
                        alpha_rad = np.deg2rad(data['alpha'][i])
                        bank_rad = np.deg2rad(data['bank'][i])
                        CL, CD, _ = self.aerodynamic_coefficients(alpha_rad, bank_rad, Ma)
                        LD = CL / CD if CD > 0 else 0
                        LD_vals.append(LD)
                    
                    # 在右侧y轴显示升阻比
                    ax_ld = ax_ctrl.twinx()
                    ld_line = ax_ld.plot(t, LD_vals, color='#9b59b6', linewidth=1.8, 
                                        linestyle=':', alpha=0.7, label='L/D Ratio')
                    ax_ld.set_ylabel('L/D Ratio', color='#9b59b6', fontsize=9)
                    ax_ld.tick_params(axis='y', labelcolor='#9b59b6', labelsize=8)
                    ax_ld.set_ylim(0, max(LD_vals) * 1.1 if LD_vals else 10)
                    
                    ax_ctrl.set_xlabel('Time (s)')
                    ax_ctrl.set_ylabel('Angle (°)')
                    # 标题已移除
                    ax_ctrl.grid(True, alpha=0.3)
                    
                    # 合并图例
                    lines1, labels1 = ax_ctrl.get_legend_handles_labels()
                    lines2, labels2 = ax_ld.get_legend_handles_labels()
                    ax_ctrl.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
                    
                    ax_ctrl.set_xlim(0, duration)
                    ax_ctrl.set_ylim(-35, 35)
                    
                    subplot_index += 1
            
            plt.tight_layout()
            # 确保目录存在并保存图片
            import os
            os.makedirs('data_generation/data_figure', exist_ok=True)
            plt.savefig('data_generation/data_figure/hgv_control_parameters_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ HGV控制参数分析图已保存到: data_generation/data_figure/hgv_control_parameters_analysis.png")
        else:
            print("没有生成用于绘图的轨迹")

    def generate_3d_trajectory_plots(self):
        """
        生成HGV 3D轨迹投影图
        展示真实的轨迹投影到不同平面，并添加底面机动轨迹线
        分别保存C型和S型机动为独立图片
        """
        # 生成HGV 3D轨迹投影图
        
        # 生成C型和S型机动（用真实的turning和weaving）- 球坐标系统
        c_initial = {'h': 65000, 'λ': 0.0, 'φ': 0.0, 'V': 6501.1, 'γ': -0.0077, 'ψ': 0.0185}
        sampling_interval_s = float(getattr(self, 'sampling_interval_s', 1.0))
        points_per_trajectory = int(getattr(self, 'points_per_trajectory', 1000))
        duration = float(getattr(self, 'custom_duration', (points_per_trajectory - 1) * sampling_interval_s))
        t_c, traj_c = self.simulate_trajectory(
            c_initial,
            'turning',
            duration,
            sampling_interval_s=sampling_interval_s,
            num_points=points_per_trajectory,
        )
        
        s_initial = {'h': 65000, 'λ': 0.0, 'φ': 0.0, 'V': 6500.8, 'γ': -0.0077, 'ψ': -0.0154}
        t_s, traj_s = self.simulate_trajectory(
            s_initial,
            'weaving',
            duration,
            sampling_interval_s=sampling_interval_s,
            num_points=points_per_trajectory,
        )
        
        if t_c is None or traj_c is None or t_s is None or traj_s is None:
            print("生成C/S轨迹失败")
            return
        
        # 直接使用球坐标数据 - 轨迹格式: [h, λ, φ, V, γ, ψ]
        # C型轨迹坐标提取
        h_c = traj_c[:, 0] / 1000  # 高度转换为km
        lon_c = traj_c[:, 1] * 180 / np.pi  # 经度转换为度
        lat_c = traj_c[:, 2] * 180 / np.pi  # 纬度转换为度
        
        # S型轨迹坐标提取
        h_s = traj_s[:, 0] / 1000  # 高度转换为km
        lon_s = traj_s[:, 1] * 180 / np.pi  # 经度转换为度
        lat_s = traj_s[:, 2] * 180 / np.pi  # 纬度转换为度
        
        # === 第一个图：C型机动 ===
        plt.style.use('default')  # 重置样式
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        
        # 计算速度用于颜色映射
        V_c = traj_c[:, 3]  # 速度大小
        V_c_normalized = (V_c - np.min(V_c)) / (np.max(V_c) - np.min(V_c))
        
        # 使用速度颜色映射的主轨迹
        colors_c = plt.cm.viridis(V_c_normalized)
        for i in range(len(lon_c)-1):
            ax1.plot(lon_c[i:i+2], lat_c[i:i+2], h_c[i:i+2],
                    color=colors_c[i], linewidth=3, alpha=0.9)
        
        # 添加颜色条
        sm_c = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                    norm=plt.Normalize(vmin=np.min(V_c), vmax=np.max(V_c)))
        sm_c.set_array([])
        cbar_c = plt.colorbar(sm_c, ax=ax1, shrink=0.6, aspect=20, pad=0.1)
        cbar_c.set_label('Velocity (m/s)', fontsize=10)
        
        # 投影线已移除
        
        # 底面轨迹投影（更透明的灰色）
        base_height = np.min(h_c) - 3
        ax1.plot(lon_c, lat_c, np.full_like(h_c, base_height), 
                color='#d3d3d3', linewidth=2, linestyle='-', alpha=0.5, label='Ground Track')
        
        # 设置坐标轴
        ax1.set_xlabel('Lambda (°)')
        ax1.set_ylabel('Phi (°)')
        ax1.set_zlabel('Height (km)')
        
        # 创建自定义图例
        from matplotlib.lines import Line2D
        legend_elements_c = [
            Line2D([0], [0], color='#d3d3d3', linewidth=2, alpha=0.5, label='Ground Track')
        ]
        ax1.legend(handles=legend_elements_c, fontsize=10, loc='upper right')
        
        # 设置透明背景
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.xaxis.pane.set_edgecolor('none')
        ax1.yaxis.pane.set_edgecolor('none')
        ax1.zaxis.pane.set_edgecolor('none')
        ax1.xaxis.pane.set_alpha(0)
        ax1.yaxis.pane.set_alpha(0)
        ax1.zaxis.pane.set_alpha(0)
        ax1.grid(True, alpha=0.15)
        
        # 设置合理的坐标轴范围
        ax1.set_xlim(np.min(lon_c) - 2, np.max(lon_c) + 2)
        ax1.set_ylim(np.min(lat_c) - 1, np.max(lat_c) + 1)
        ax1.set_zlim(base_height, np.max(h_c) + 2)
        ax1.view_init(elev=25, azim=45)
        
        # 保存C型机动图
        plt.tight_layout()
        import os
        os.makedirs('data_generation/data_figure', exist_ok=True)
        plt.savefig('data_generation/data_figure/hgv_3d_trajectory_c_maneuver.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='white', pad_inches=0.1)
        plt.close()
        
        # === 第二个图：S型机动 ===
        plt.style.use('default')  # 重置样式
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        # 计算速度用于颜色映射
        V_s = traj_s[:, 3]  # 速度大小
        V_s_normalized = (V_s - np.min(V_s)) / (np.max(V_s) - np.min(V_s))
        
        # 使用速度颜色映射的主轨迹
        colors_s = plt.cm.viridis(V_s_normalized)
        for i in range(len(lon_s)-1):
            ax2.plot(lon_s[i:i+2], lat_s[i:i+2], h_s[i:i+2],
                    color=colors_s[i], linewidth=3, alpha=0.9)
        
        # 添加颜色条
        sm_s = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                    norm=plt.Normalize(vmin=np.min(V_s), vmax=np.max(V_s)))
        sm_s.set_array([])
        cbar_s = plt.colorbar(sm_s, ax=ax2, shrink=0.6, aspect=20, pad=0.1)
        cbar_s.set_label('Velocity (m/s)', fontsize=10)
        
        # 投影线已移除
        
        # 底面轨迹投影（更透明的灰色）
        base_height_s = np.min(h_s) - 3
        ax2.plot(lon_s, lat_s, np.full_like(h_s, base_height_s), 
                color='#d3d3d3', linewidth=2, linestyle='-', alpha=0.5, label='Ground Track')
        
        # 设置坐标轴
        ax2.set_xlabel('Lambda (°)')
        ax2.set_ylabel('Phi (°)')
        ax2.set_zlabel('Height (km)')
        
        # 创建自定义图例
        legend_elements_s = [
            Line2D([0], [0], color='#d3d3d3', linewidth=2, alpha=0.5, label='Ground Track')
        ]
        ax2.legend(handles=legend_elements_s, fontsize=10, loc='upper right')
        
        # 设置透明背景
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor('none')
        ax2.yaxis.pane.set_edgecolor('none')
        ax2.zaxis.pane.set_edgecolor('none')
        ax2.xaxis.pane.set_alpha(0)
        ax2.yaxis.pane.set_alpha(0)
        ax2.zaxis.pane.set_alpha(0)
        ax2.grid(True, alpha=0.15)
        
        # 设置合理的坐标轴范围 - 优化S型显示
        ax2.set_xlim(np.min(lon_s) - 2, np.max(lon_s) + 3)  # 增加X轴范围以显示投影
        ax2.set_ylim(np.min(lat_s) - 1, np.max(lat_s) + 1)
        ax2.set_zlim(base_height_s, np.max(h_s) + 2)
        ax2.view_init(elev=25, azim=45)
        
        # 保存S型机动图
        plt.tight_layout()
        import os
        os.makedirs('data_generation/data_figure', exist_ok=True)
        plt.savefig('data_generation/data_figure/hgv_3d_trajectory_s_maneuver.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='white', pad_inches=0.1)
        plt.close()
        
        print(f"✅ HGV 3D轨迹图已分别保存:")
        print(f"   - C型机动: data_generation/data_figure/hgv_3d_trajectory_c_maneuver.png")
        print(f"   - S型机动: data_generation/data_figure/hgv_3d_trajectory_s_maneuver.png")

    def generate_dataset_diversity_figures(self, trajectories, n_samples_per_type=3):
        """
        生成数据集多样性的四张独立图片 - 顶刊标准
        (a) 3D轨迹空间分布
        (b) 操作包络覆盖
        (c) 地理覆盖范围
        (d) 初始条件统计
        """
        print(f"📊 生成数据集多样性可视化 (4张独立图片)...")
        
        # 统一绘图风格配置
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['legend.framealpha'] = 0.9
        
        # 统一配色方案
        COLORS = {
            'longitudinal': '#1f77b4',  # 蓝色
            'turning': '#d62728',       # 红色
            'weaving': '#2ca02c'        # 绿色
        }
        
        MARKERS = {
            'longitudinal': 'o',  # 圆形
            'turning': 's',       # 方形
            'weaving': '^'        # 三角形
        }
        
        # 确保输出目录存在
        os.makedirs('data_generation/data_figure', exist_ok=True)
        
        # 生成四张图
        self._plot_trajectory_distribution(trajectories, n_samples_per_type, COLORS, MARKERS)
        self._plot_operational_envelope(trajectories, COLORS, MARKERS)
        self._plot_geographical_coverage(trajectories, COLORS, MARKERS)
        self._plot_initial_conditions_stats(trajectories, COLORS)
        
        print(f"✅ 数据集多样性可视化完成！")
        print(f"    - dataset_trajectory_distribution.png/pdf")
        print(f"    - dataset_operational_envelope.png/pdf")
        print(f"    - dataset_geographical_coverage.png/pdf")
        print(f"    - dataset_initial_conditions.png/pdf")
    
    def _plot_trajectory_distribution(self, trajectories, n_samples_per_type, COLORS, MARKERS):
        """
        (a) 3D轨迹空间分布 - 展示三种机动类型的典型轨迹形态
        优化: 减少轨迹数量(6条)，添加起点/终点标记，图例放外部
        """
        print("  📈 (a) 生成3D轨迹空间分布图...")
        
        # 按机动类型分组
        maneuver_groups = {'longitudinal': [], 'turning': [], 'weaving': []}
        for traj in trajectories:
            maneuver_type = traj['maneuver_type']
            if maneuver_type in maneuver_groups:
                maneuver_groups[maneuver_type].append(traj)
        
        # 每种类型选择2条（减少拥挤）
        n_samples = 2
        selected_trajectories = []
        for maneuver_type in ['longitudinal', 'turning', 'weaving']:
            selected_trajectories.extend(maneuver_groups[maneuver_type][:n_samples])
        
        # 创建3D图
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 收集起点和终点用于标记
        start_points = {'longitudinal': [], 'turning': [], 'weaving': []}
        end_points = {'longitudinal': [], 'turning': [], 'weaving': []}
        
        # 绘制轨迹
        for traj in selected_trajectories:
            maneuver_type = traj['maneuver_type']
            
            # 提取球坐标数据
            λ_coords = np.rad2deg(traj['λ'])  # 经度
            φ_coords = np.rad2deg(traj['φ'])  # 纬度
            h_coords = (traj['r'] - self.R_earth) / 1000  # 高度km
            V_coords = traj['V']  # 速度
            
            # 速度归一化用于颜色映射
            V_normalized = (V_coords - np.min(V_coords)) / (np.max(V_coords) - np.min(V_coords) + 1e-8)
            
            # 使用速度颜色映射
            colors = plt.cm.viridis(V_normalized)
            for i in range(len(λ_coords)-1):
                ax.plot(λ_coords[i:i+2], φ_coords[i:i+2], h_coords[i:i+2],
                       color=colors[i], linewidth=2.8, alpha=0.9)
            
            # 收集起点和终点
            start_points[maneuver_type].append((λ_coords[0], φ_coords[0], h_coords[0]))
            end_points[maneuver_type].append((λ_coords[-1], φ_coords[-1], h_coords[-1]))
        
        # 绘制起点标记（大圆圈）
        for maneuver_type in ['longitudinal', 'turning', 'weaving']:
            if start_points[maneuver_type]:
                starts = np.array(start_points[maneuver_type])
                ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2],
                          c=COLORS[maneuver_type], marker='o', s=100,
                          edgecolors='black', linewidths=2, alpha=0.95,
                          zorder=100)
        
        # 绘制终点标记（方形）
        for maneuver_type in ['longitudinal', 'turning', 'weaving']:
            if end_points[maneuver_type]:
                ends = np.array(end_points[maneuver_type])
                ax.scatter(ends[:, 0], ends[:, 1], ends[:, 2],
                          c=COLORS[maneuver_type], marker='s', s=80,
                          edgecolors='black', linewidths=1.5, alpha=0.9,
                          zorder=100)
        
        # 添加速度colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                   norm=plt.Normalize(vmin=5000, vmax=7000))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.15)
        cbar.set_label('Velocity (m/s)', fontsize=11)
        
        # 创建图例 - 放在外部右侧，避免遮挡数据
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=COLORS['longitudinal'], linewidth=3, label='Longitudinal'),
            Line2D([0], [0], color=COLORS['turning'], linewidth=3, label='Turning (C-shaped)'),
            Line2D([0], [0], color=COLORS['weaving'], linewidth=3, label='Weaving (S-shaped)'),
            Line2D([0], [0], marker='o', color='gray', markersize=8, 
                   linestyle='None', markeredgecolor='black', markeredgewidth=1.5, label='Start'),
            Line2D([0], [0], marker='s', color='gray', markersize=7, 
                   linestyle='None', markeredgecolor='black', markeredgewidth=1.5, label='End')
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='upper left',
                 bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        
        # 设置坐标轴（使用LaTeX格式统一样式）
        ax.set_xlabel(r'$\lambda$ (°)', fontsize=12)
        ax.set_ylabel(r'$\phi$ (°)', fontsize=12)
        ax.set_zlabel(r'$h$ (km)', fontsize=12)
        ax.grid(True, alpha=0.15)
        
        # 设置透明背景
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # 设置视角
        ax.view_init(elev=25, azim=45)
        
        # 保存
        plt.tight_layout()
        plt.savefig('data_generation/data_figure/dataset_trajectory_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig('data_generation/data_figure/dataset_trajectory_distribution.pdf', 
                   bbox_inches='tight')
        plt.close()
        print("      ✓ dataset_trajectory_distribution.png/pdf")
    
    def _plot_operational_envelope(self, trajectories, COLORS, MARKERS):
        """
        (b) 操作包络覆盖 - 展示高度-速度空间的覆盖范围
        优化: 改为1×3分面布局 + 2D密度图，清晰展示每种机动类型的包络特征
        """
        print("  📈 (b) 生成操作包络覆盖图...")
        
        # 创建1×3子图布局
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        maneuver_types = ['longitudinal', 'turning', 'weaving']
        maneuver_names = ['Longitudinal', 'Turning (C-shaped)', 'Weaving (S-shaped)']
        
        for idx, (maneuver_type, name) in enumerate(zip(maneuver_types, maneuver_names)):
            ax = axes[idx]
            
            # 收集当前机动类型的数据
            h_all = []
            v_all = []
            
            for traj in trajectories:
                if traj['maneuver_type'] == maneuver_type:
                    h = (traj['r'] - self.R_earth) / 1000  # 高度km
                    v = traj['V'] / 1000  # 速度km/s
                    # 降采样：每10个点取1个，减少密度
                    h_all.extend(h[::10])
                    v_all.extend(v[::10])
            
            # 绘制2D密度图（hexbin）
            hexbin = ax.hexbin(v_all, h_all, 
                              gridsize=30, 
                              cmap='YlOrRd', 
                              mincnt=1,
                              alpha=0.8,
                              edgecolors='none')
            
            # 添加colorbar（只在最右侧子图）
            if idx == 2:
                cbar = plt.colorbar(hexbin, ax=ax, pad=0.02)
                cbar.set_label('Sample Count', fontsize=10)
            
            # 叠加降采样散点图，显示分布边界
            ax.scatter(v_all, h_all, 
                      c=COLORS[maneuver_type], 
                      marker=MARKERS[maneuver_type],
                      s=3, alpha=0.15, 
                      edgecolors='none',
                      zorder=1)
            
            # 添加典型操作区域框（只在中间子图）
            if idx == 1:
                from matplotlib.patches import Rectangle
                rect = Rectangle((5.0, 35), 2.0, 35, 
                                linewidth=2, edgecolor='gray', 
                                facecolor='none', linestyle='--', alpha=0.6)
                ax.add_patch(rect)
                ax.text(6.0, 68, 'Hypersonic Glide Regime', 
                       fontsize=9, ha='center', style='italic',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='wheat', alpha=0.4))
            
            # 设置坐标轴
            ax.set_xlabel('V (km/s)', fontsize=11)
            if idx == 0:
                ax.set_ylabel('h (km)', fontsize=11)
            ax.set_xlim(4.5, 7.5)
            ax.set_ylim(35, 75)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
            
            # 移除子图内部标签，避免与整体Figure的(a)(b)(c)(d)冲突
            # 改为在顶部中央显示机动类型
            ax.text(0.5, 0.97, name, 
                   transform=ax.transAxes, fontsize=11, fontweight='bold',
                   horizontalalignment='center', verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor=COLORS[maneuver_type], alpha=0.25))
        
        # 保存
        plt.tight_layout()
        plt.savefig('data_generation/data_figure/dataset_operational_envelope.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig('data_generation/data_figure/dataset_operational_envelope.pdf', 
                   bbox_inches='tight')
        plt.close()
        print("      ✓ dataset_operational_envelope.png/pdf")
    
    def _plot_geographical_coverage(self, trajectories, COLORS, MARKERS):
        """
        (c) 地理覆盖范围 - 展示经度-纬度投影和轨迹终点分布
        优化: 增加轨迹可见性，展示终点分布的多样性
        """
        print("  📈 (c) 生成地理覆盖范围图...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制轨迹线（增加可见性）
        for traj in trajectories:
            maneuver_type = traj['maneuver_type']
            λ_coords = np.rad2deg(traj['λ'])
            φ_coords = np.rad2deg(traj['φ'])
            
            ax.plot(λ_coords, φ_coords, 
                   color=COLORS[maneuver_type], 
                   linewidth=1.0, alpha=0.35)  # 增加透明度和线宽
        
        # 绘制起点标记（小标记，不是重点）
        np.random.seed(42)
        for maneuver_type in ['longitudinal', 'turning', 'weaving']:
            λ_start = []
            φ_start = []
            
            for traj in trajectories:
                if traj['maneuver_type'] == maneuver_type:
                    λ_start.append(np.rad2deg(traj['λ'][0]))
                    φ_start.append(np.rad2deg(traj['φ'][0]))
            
            # 轻微抖动
            jitter = 0.15
            λ_start_jitter = np.array(λ_start) + np.random.uniform(-jitter, jitter, len(λ_start))
            φ_start_jitter = np.array(φ_start) + np.random.uniform(-jitter, jitter, len(φ_start))
            
            ax.scatter(λ_start_jitter, φ_start_jitter, 
                      c=COLORS[maneuver_type], 
                      marker='o',
                      s=50, alpha=0.4,  # 小标记，低透明度
                      edgecolors='black', linewidths=0.5,
                      zorder=8)
        
        # 绘制终点标记（大标记，突出多样性）
        np.random.seed(43)  # 不同的随机种子
        for maneuver_type in ['longitudinal', 'turning', 'weaving']:
            λ_end = []
            φ_end = []
            
            for traj in trajectories:
                if traj['maneuver_type'] == maneuver_type:
                    λ_end.append(np.rad2deg(traj['λ'][-1]))
                    φ_end.append(np.rad2deg(traj['φ'][-1]))
            
            # 轻微抖动
            jitter = 0.2
            λ_end_jitter = np.array(λ_end) + np.random.uniform(-jitter, jitter, len(λ_end))
            φ_end_jitter = np.array(φ_end) + np.random.uniform(-jitter, jitter, len(φ_end))
            
            ax.scatter(λ_end_jitter, φ_end_jitter, 
                      c=COLORS[maneuver_type], 
                      marker=MARKERS[maneuver_type],
                      s=140, alpha=0.85,  # 大标记，高透明度
                      edgecolors='black', linewidths=1.8,
                      label=f'{maneuver_type.capitalize()}',
                      zorder=10)
        
        # 添加原点标记
        ax.scatter([0], [0], marker='x', s=250, c='red', 
                  linewidths=3.5, zorder=5, label='Origin (0°, 0°)')
        
        # 添加比例尺
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        scale_length = 10
        scale_x = xlim[1] - 12
        scale_y = ylim[0] + 3
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
               'k-', linewidth=3, solid_capstyle='butt')
        ax.plot([scale_x, scale_x], [scale_y - 0.5, scale_y + 0.5], 'k-', linewidth=2)
        ax.plot([scale_x + scale_length, scale_x + scale_length], 
               [scale_y - 0.5, scale_y + 0.5], 'k-', linewidth=2)
        ax.text(scale_x + scale_length/2, scale_y + 1.5, f'{scale_length}°', 
               ha='center', fontsize=10, fontweight='bold')
        
        # 设置坐标轴（使用LaTeX格式）
        ax.set_xlabel(r'$\lambda$ (°)', fontsize=12)
        ax.set_ylabel(r'$\phi$ (°)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right', framealpha=0.9, 
                 title='Trajectory Endpoints', title_fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        
        # 保存
        plt.tight_layout()
        plt.savefig('data_generation/data_figure/dataset_geographical_coverage.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig('data_generation/data_figure/dataset_geographical_coverage.pdf', 
                   bbox_inches='tight')
        plt.close()
        print("      ✓ dataset_geographical_coverage.png/pdf")
    
    def _plot_initial_conditions_stats(self, trajectories, COLORS):
        """
        (d) 初始条件统计分布 - 展示系统化的初始条件变化
        使用2x2分面布局，每个参数独立显示
        """
        print("  📈 (d) 生成初始条件统计分布图...")
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # 收集初始条件数据
        maneuver_types = ['longitudinal', 'turning', 'weaving']
        initial_data = {mt: {'h0': [], 'V0': [], 'γ0': [], 'ψ0': []} for mt in maneuver_types}
        
        for traj in trajectories:
            mt = traj['maneuver_type']
            if mt in initial_data:
                initial_data[mt]['h0'].append((traj['r'][0] - self.R_earth) / 1000)  # km
                initial_data[mt]['V0'].append(traj['V'][0] / 1000)  # km/s
                initial_data[mt]['γ0'].append(np.rad2deg(traj['γ'][0]))  # degree
                initial_data[mt]['ψ0'].append(np.rad2deg(traj['ψ'][0]))  # degree
        
        # 参数配置
        param_configs = [
            {'key': 'h0', 'label': 'h₀ (km)', 'ylabel': 'Altitude (km)'},
            {'key': 'V0', 'label': 'V₀ (km/s)', 'ylabel': 'Velocity (km/s)'},
            {'key': 'γ0', 'label': 'γ₀ (°)', 'ylabel': 'Flight Path Angle (°)'},
            {'key': 'ψ0', 'label': 'ψ₀ (°)', 'ylabel': 'Heading Angle (°)'}
        ]
        
        # 为每个参数绘制独立的箱线图
        for idx, config in enumerate(param_configs):
            ax = axes[idx]
            param_key = config['key']
            
            # 准备数据
            data_to_plot = [initial_data[mt][param_key] for mt in maneuver_types]
            positions = np.arange(len(maneuver_types))
            
            # 绘制箱线图
            bp = ax.boxplot(data_to_plot,
                           positions=positions,
                           widths=0.6,
                           patch_artist=True,
                           tick_labels=[mt.capitalize() for mt in maneuver_types],
                           medianprops=dict(color='black', linewidth=2.5),
                           showfliers=True)
            
            # 设置颜色
            for patch, mt in zip(bp['boxes'], maneuver_types):
                patch.set_facecolor(COLORS[mt])
                patch.set_alpha(0.7)
            
            for whisker, cap, mt in zip(bp['whiskers'], bp['caps'], 
                                       maneuver_types * 2):
                whisker.set_color(COLORS[mt])
                whisker.set_linewidth(1.5)
                cap.set_color(COLORS[mt])
                cap.set_linewidth(1.5)
            
            # 设置标签和网格（完全移除子图标题，符合顶刊标准）
            ax.set_ylabel(config['ylabel'], fontsize=11)
            # 移除子图标题，只在caption中描述
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=15)
            
            # 在子图内部添加参数标签（左上角）
            ax.text(0.02, 0.98, config['label'], 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
            
            # 优化Y轴范围，展示细微差异
            if param_key == 'γ0':
                # 航迹角范围较小，扩大显示范围
                y_min, y_max = ax.get_ylim()
                y_center = (y_min + y_max) / 2
                y_range = max(0.2, y_max - y_min)  # 至少0.2度范围
                ax.set_ylim(y_center - y_range*0.6, y_center + y_range*0.6)
            elif param_key == 'ψ0':
                # 航向角范围较小，扩大显示范围
                y_min, y_max = ax.get_ylim()
                y_center = (y_min + y_max) / 2
                y_range = max(0.15, y_max - y_min)  # 至少0.15度范围
                ax.set_ylim(y_center - y_range*0.7, y_center + y_range*0.7)
            
            # 添加统计信息（均值标记）
            for i, mt in enumerate(maneuver_types):
                mean_val = np.mean(initial_data[mt][param_key])
                ax.plot(i, mean_val, marker='D', color='red', 
                       markersize=6, zorder=10, alpha=0.8)
        
        # 添加总体统计信息
        stats_text = f"Dataset: {len(trajectories)} trajectories | "
        for mt in maneuver_types:
            count = len([t for t in trajectories if t['maneuver_type'] == mt])
            stats_text += f"{mt.capitalize()}: {count} | "
        fig.text(0.5, 0.02, stats_text.strip(' | '), 
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 添加图例说明
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=2.5, label='Median'),
            Line2D([0], [0], marker='D', color='red', markersize=6, 
                  linestyle='None', label='Mean')
        ]
        fig.legend(handles=legend_elements, loc='upper right', 
                  fontsize=10, framealpha=0.9)
        
        # 保存
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig('data_generation/data_figure/dataset_initial_conditions.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig('data_generation/data_figure/dataset_initial_conditions.pdf', 
                   bbox_inches='tight')
        plt.close()
        print("      ✓ dataset_initial_conditions.png/pdf")

def save_trajectories_to_file(trajectories, filepath):
    """
    将轨迹保存到NPZ文件
    """
    print(f"保存 {len(trajectories)} 条轨迹到 {filepath}")
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare data for saving
    trajectory_data = []
    maneuver_types = []
    
    for traj in trajectories:
        trajectory_data.append({
            'r': traj['r'],
            'λ': traj['λ'], 
            'φ': traj['φ'],
            'V': traj['V'],
            'γ': traj['γ'],
            'ψ': traj['ψ'],
            'time': traj['time'],
            'alpha': traj['alpha'],
            'bank': traj['bank'],
            'velocity': traj['velocity']
        })
        maneuver_types.append(traj['maneuver_type'])
    
    # Save to NPZ file
    np.savez_compressed(filepath, 
                       trajectories=trajectory_data,
                       maneuver_types=maneuver_types)
    
    print(f"轨迹已成功保存到 {filepath}")

def test_single_trajectory(duration=999.0, sampling_interval_s=1.0, points_per_trajectory=1000):
    """
    测试改进后的高精度轨迹生成 - 球坐标系统
    """
    print("测试改进的高精度轨迹生成...")
    print("=" * 70)
    
    simulator = DCBNN_HGV_Simulator()
    
    # 使用DCBNN论文Table 1的严格标准初始条件 - 球坐标系
    initial_conditions = {
        'h': 65000,                             # H₀ = 65km (DCBNN Table 1)
        'λ': 0.0,                               # 初始经度 = 0
        'φ': 0.0,                               # 初始纬度 = 0
        'V': 6500,                              # 初始速度大小 (m/s)
        'γ': np.deg2rad(-0.44),                 # 初始航迹角 (rad)
        'ψ': np.pi/2 + np.random.uniform(-0.05, 0.05)  # 微调随机航向角
    }
    
    # 测试三种机动类型
    maneuver_types = ['longitudinal', 'turning', 'weaving']
    duration = float(duration)
    sampling_interval_s = float(sampling_interval_s)
    points_per_trajectory = int(points_per_trajectory)
    
    print(f"测试参数:")
    print(f"    Duration: {duration} seconds")
    print(f"    Sampling: {sampling_interval_s:g}-second interval")
    print(f"    Integration: RK45 with rtol=1e-6, atol=1e-8")
    print(f"    Expected points per trajectory: {points_per_trajectory}")
    print("=" * 70)
    
    all_successful = True
    results = {}
    
    for maneuver_type in maneuver_types:
        print(f"测试 {maneuver_type} 机动...")
        
        # 仿真
        simulator.sampling_interval_s = sampling_interval_s
        simulator.points_per_trajectory = points_per_trajectory
        t, trajectory = simulator.simulate_trajectory(
            initial_conditions,
            maneuver_type,
            duration,
            sampling_interval_s=sampling_interval_s,
            num_points=points_per_trajectory,
        )
        
        if t is not None and trajectory is not None:
            # 球坐标轨迹分析 [r, λ, φ, V, γ, ψ]
            r_coords = trajectory[:, 0]  # 地心距离
            λ_coords = trajectory[:, 1]  # 经度
            φ_coords = trajectory[:, 2]  # 纬度
            V_coords = trajectory[:, 3]  # 速度大小
            γ_coords = trajectory[:, 4]  # 航迹角
            ψ_coords = trajectory[:, 5]  # 航向角
            
            # 计算物理量
            r_change = (trajectory[-1, 0] - trajectory[0, 0]) / 1000  # km
            h_change = r_change  # 地心距离变化等于高度变化
            λ_range = np.rad2deg(np.max(λ_coords) - np.min(λ_coords))  # 度
            φ_range = np.rad2deg(np.max(φ_coords) - np.min(φ_coords))  # 度
            
            V_start = trajectory[0, 3]
            V_end = trajectory[-1, 3]
            
            # 计算马赫数 (使用地心距离计算高度)
            h_start = trajectory[0, 0] - simulator.R_earth
            h_end = trajectory[-1, 0] - simulator.R_earth
            rho, a = simulator.atmospheric_model(h_start)
            ma_start = V_start / a
            rho_end, a_end = simulator.atmospheric_model(h_end)
            ma_end = V_end / a_end
            
            results[maneuver_type] = {
                'points': len(t),
                'λ_range': λ_range,      # 度
                'φ_range': φ_range,      # 度  
                'h_change': h_change,    # km
                'v_start': V_start / 1000,       # km/s
                'v_end': V_end / 1000,           # km/s
                'ma_start': ma_start,
                'ma_end': ma_end
            }
            
            print(f"  {maneuver_type} 成功:")
            print(f"      生成点数: {len(t)}")
            print(f"      经度范围: {λ_range:.3f}°")
            print(f"      纬度范围: {φ_range:.3f}°")
            print(f"      高度变化: {h_change:.1f} km")
            print(f"      速度: {V_start/1000:.2f} → {V_end/1000:.2f} km/s")
            print(f"      马赫数: {ma_start:.1f} → {ma_end:.1f}")
            
            # 验证机动特征
            if maneuver_type == 'longitudinal' and abs(φ_range) > 0.5:
                print(f"      ⚠️ Warning: Longitudinal maneuver has significant lateral motion ({φ_range:.3f}°)")
            elif maneuver_type == 'turning' and abs(λ_range) < 0.1:
                print(f"      ⚠️ Warning: Turning maneuver has limited lateral motion ({λ_range:.3f}°)")
            elif maneuver_type == 'weaving' and abs(λ_range) < 0.2:
                print(f"      ⚠️ Warning: Weaving maneuver has limited lateral motion ({λ_range:.3f}°)")
                
        else:
            print(f"  {maneuver_type} 失败!")
            all_successful = False
            
    print("\n" + "=" * 70)
    if all_successful:
        print("所有轨迹测试通过!")
        print("\n改进验证总结:")
        print("    ✅ 球坐标动力学方程")
        print("    ✅ 高精度数值积分 (RK45)")
        print(f"    ✅ {sampling_interval_s:g}s sampling interval")
        print("    ✅ 简化的气动系数")
        print("    ✅ 清晰的控制策略")
        print("    ✅ 全面的物理验证")
        
        print("\n轨迹特征:")
        for maneuver, data in results.items():
            print(f"    {maneuver:12s}: λ={data['λ_range']:6.3f}°, φ={data['φ_range']:6.3f}°, Δh={data['h_change']:5.1f}km")
        
        return True
    else:
        print("部分轨迹测试失败!")
        return False

def main():
    """
    简化的主函数 - 直接运行数据生成和可视化
    """
    from utils.trajectory_protocol import (
        DEFAULT_POINTS_PER_TRAJECTORY,
        DEFAULT_PRED_LEN,
        DEFAULT_SAMPLING_INTERVAL_S,
        DEFAULT_SEQ_LEN,
        DEFAULT_TRAJECTORY_COUNT,
        DEFAULT_WINDOW_STRIDE,
    )

    # ===== 用户可调整参数 =====
    TRAJECTORY_COUNT = int(os.getenv("HGV_TRAJECTORY_COUNT", str(DEFAULT_TRAJECTORY_COUNT)))
    SAMPLING_INTERVAL_S = float(os.getenv("HGV_SAMPLING_INTERVAL_S", str(DEFAULT_SAMPLING_INTERVAL_S)))
    POINTS_PER_TRAJECTORY = int(os.getenv("HGV_POINTS_PER_TRAJECTORY", str(DEFAULT_POINTS_PER_TRAJECTORY)))
    if "HGV_SIMULATION_DURATION" in os.environ and "HGV_POINTS_PER_TRAJECTORY" not in os.environ:
        SIMULATION_DURATION = float(os.getenv("HGV_SIMULATION_DURATION", "999"))
        POINTS_PER_TRAJECTORY = int(round(SIMULATION_DURATION / SAMPLING_INTERVAL_S)) + 1
    else:
        SIMULATION_DURATION = (POINTS_PER_TRAJECTORY - 1) * SAMPLING_INTERVAL_S
    SEQ_LEN = int(os.getenv("HGV_SEQ_LEN", str(DEFAULT_SEQ_LEN)))
    PRED_LEN = int(os.getenv("HGV_PRED_LEN", str(DEFAULT_PRED_LEN)))
    WINDOW_STRIDE = int(os.getenv("HGV_WINDOW_STRIDE", str(DEFAULT_WINDOW_STRIDE)))
    DATA_SEED = int(os.getenv("HGV_DATA_SEED", "42"))
    SKIP_DATA_FIGURES = os.getenv("HGV_SKIP_DATA_FIGURES", "0").strip().lower() in {"1", "true", "yes", "on"}
    PROCESSED_DATA_DIR = str(get_processed_data_dir(PROJECT_ROOT))
    if SEQ_LEN <= 0 or PRED_LEN <= 0 or WINDOW_STRIDE <= 0 or POINTS_PER_TRAJECTORY <= 1 or SAMPLING_INTERVAL_S <= 0:
        raise ValueError("HGV sequence/window settings, points_per_trajectory, and sampling interval must be positive.")
    # =========================
    
    np.random.seed(DATA_SEED)
    print("HGV轨迹仿真器")
    print(f"设置: {TRAJECTORY_COUNT} trajectories, {POINTS_PER_TRAJECTORY} points/trajectory, {SAMPLING_INTERVAL_S:g}s sampling, seed={DATA_SEED}")
    print(f"窗口: seq_len={SEQ_LEN} ({SEQ_LEN * SAMPLING_INTERVAL_S:g}s), pred_len={PRED_LEN} ({PRED_LEN * SAMPLING_INTERVAL_S:g}s), stride={WINDOW_STRIDE}")
    
    simulator = DCBNN_HGV_Simulator()
    simulator.sampling_interval_s = SAMPLING_INTERVAL_S
    simulator.points_per_trajectory = POINTS_PER_TRAJECTORY
    # 设置自定义仿真时长
    simulator.custom_duration = SIMULATION_DURATION
    
    # 1. 快速测试
    success = test_single_trajectory(
        duration=SIMULATION_DURATION,
        sampling_interval_s=SAMPLING_INTERVAL_S,
        points_per_trajectory=POINTS_PER_TRAJECTORY,
    )
    
    if not success:
        print("测试失败，程序终止")
        return
    
    # 2. 生成数据集
    print(f"生成数据集 ({TRAJECTORY_COUNT}条轨迹)...")
    
    # 临时修改生成数量
    original_count = TRAJECTORY_COUNT
    trajectories = simulator.generate_trajectories(original_count)
    
    if len(trajectories) == 0:
        print("❌ 数据集生成失败")
        return
    
    print(f"成功生成 {len(trajectories)} 条轨迹")
    raw_trajectories_path = get_raw_trajectories_npz_path(PROJECT_ROOT)
    save_raw_trajectories(
        trajectories,
        raw_trajectories_path,
        sampling_interval_s=SAMPLING_INTERVAL_S,
        points_per_trajectory=POINTS_PER_TRAJECTORY,
        trajectory_duration_s=SIMULATION_DURATION,
        generation_seed=DATA_SEED,
    )
    print(f"    ✓ 原始长轨迹已保存: {raw_trajectories_path}")
    
    # 3. 保存数据
    
    # 转换为PIT格式
    times = []
    labels = []
    trajectory_arrays = []
    
    for traj in trajectories:
        # 球坐标系统数据转换 [r, λ, φ, V, γ, ψ]
        traj_array = np.column_stack([
            traj['r'],      # 地心距离
            traj['λ'],      # 经度
            traj['φ'],      # 纬度
            traj['V'],      # 速度
            traj['γ'],      # 航迹角
            traj['ψ']       # 航向角
        ])
        trajectory_arrays.append(traj_array)
        times.append(traj['time'])
        labels.append(traj['maneuver_type'])
    
    # 轨迹级划分：先按完整 trajectory id 分 train/val/test，再在各 split 内滑窗。
    # 这与 Autoformer/FEDformer 的“先隔离时间块，再切窗口”思想一致，
    # 但针对 HGV 多条轨迹场景改成“先隔离轨迹，再切窗口”。
    from utils.trajectory_protocol import (
        TRAJECTORY_LEVEL_PROTOCOL,
        build_windows_for_trajectories,
        split_trajectory_ids,
        validate_disjoint_trajectory_splits,
    )

    labels_array = np.asarray(labels)
    all_trajectory_ids = np.arange(len(trajectory_arrays), dtype=np.int64)
    split_ids = split_trajectory_ids(
        labels_array,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=DATA_SEED,
    )
    split_report = validate_disjoint_trajectory_splits(
        split_ids["train"],
        split_ids["val"],
        split_ids["test"],
    )
    if not split_report["is_disjoint"]:
        raise RuntimeError(f"Trajectory-level split is not disjoint: {split_report}")

    def _select(items, ids):
        return [items[int(idx)] for idx in ids]

    train_windows = build_windows_for_trajectories(
        trajectories=_select(trajectory_arrays, split_ids["train"]),
        labels=labels_array[split_ids["train"]],
        trajectory_ids=all_trajectory_ids[split_ids["train"]],
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        stride=WINDOW_STRIDE,
    )
    val_windows = build_windows_for_trajectories(
        trajectories=_select(trajectory_arrays, split_ids["val"]),
        labels=labels_array[split_ids["val"]],
        trajectory_ids=all_trajectory_ids[split_ids["val"]],
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        stride=WINDOW_STRIDE,
    )
    test_windows = build_windows_for_trajectories(
        trajectories=_select(trajectory_arrays, split_ids["test"]),
        labels=labels_array[split_ids["test"]],
        trajectory_ids=all_trajectory_ids[split_ids["test"]],
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        stride=WINDOW_STRIDE,
    )

    X_train, y_train = train_windows["X"], train_windows["y"]
    X_val, y_val = val_windows["X"], val_windows["y"]
    X_test, y_test = test_windows["X"], test_windows["y"]

    if len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0:
        # 简化的数据保存
        print(f"    保存数据到 {PROCESSED_DATA_DIR}/...")
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        all_window_labels = np.concatenate([
            train_windows["maneuver_labels"],
            val_windows["maneuver_labels"],
            test_windows["maneuver_labels"],
        ])
        split_train_indices = np.arange(len(X_train), dtype=np.int64)
        split_val_indices = np.arange(len(X_train), len(X_train) + len(X_val), dtype=np.int64)
        split_test_indices = np.arange(len(X_train) + len(X_val), len(all_window_labels), dtype=np.int64)

        # 保存数据
        dataset_path = os.path.join(PROCESSED_DATA_DIR, "hgv_trajectory_dataset.npz")
        input_scaler_path = os.path.join(PROCESSED_DATA_DIR, "scaler_hgv_trajectory.joblib")
        output_scaler_path = os.path.join(PROCESSED_DATA_DIR, "output_scaler_hgv_trajectory.joblib")
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        np.savez(dataset_path,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                dataset_protocol=np.asarray(TRAJECTORY_LEVEL_PROTOCOL),
                seq_len=np.asarray(SEQ_LEN, dtype=np.int64),
                pred_len=np.asarray(PRED_LEN, dtype=np.int64),
                window_stride=np.asarray(WINDOW_STRIDE, dtype=np.int64),
                sampling_interval_s=np.asarray(SAMPLING_INTERVAL_S, dtype=np.float64),
                points_per_trajectory=np.asarray(POINTS_PER_TRAJECTORY, dtype=np.int64),
                trajectory_duration_s=np.asarray(SIMULATION_DURATION, dtype=np.float64),
                generation_seed=np.asarray(DATA_SEED, dtype=np.int64),
                input_duration_s=np.asarray(SEQ_LEN * SAMPLING_INTERVAL_S, dtype=np.float64),
                prediction_duration_s=np.asarray(PRED_LEN * SAMPLING_INTERVAL_S, dtype=np.float64),
                maneuver_taxonomy=np.asarray(['longitudinal', 'turning', 'weaving']),
                trajectory_point_counts=np.asarray([len(t) for t in times], dtype=np.int64),
                maneuver_labels=all_window_labels,
                maneuver_labels_train=train_windows["maneuver_labels"],
                maneuver_labels_val=val_windows["maneuver_labels"],
                maneuver_labels_test=test_windows["maneuver_labels"],
                trajectory_ids_train=train_windows["trajectory_ids"],
                trajectory_ids_val=val_windows["trajectory_ids"],
                trajectory_ids_test=test_windows["trajectory_ids"],
                window_starts_train=train_windows["window_starts"],
                window_starts_val=val_windows["window_starts"],
                window_starts_test=test_windows["window_starts"],
                split_train_indices=split_train_indices,
                split_val_indices=split_val_indices,
                split_test_indices=split_test_indices,
                trajectory_split_train_ids=split_ids["train"],
                trajectory_split_val_ids=split_ids["val"],
                trajectory_split_test_ids=split_ids["test"],
                trajectory_labels=labels_array)
        
        print(f"    ✅ 数据集已保存")
        print(f"        数据协议: {TRAJECTORY_LEVEL_PROTOCOL}")
        print(f"        轨迹划分: train={len(split_ids['train'])}, val={len(split_ids['val'])}, test={len(split_ids['test'])}")
        print(f"        训练集: {X_train.shape}")
        print(f"        验证集: {X_val.shape}")  
        print(f"        测试集: {X_test.shape}")
        print(f"        轨迹重叠检查: {split_report}")
        
        # 创建并保存基础标准化器（用于后续消融实验）
        print("🔧 创建标准化器...")
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # 输入数据标准化器（仅使用训练集拟合，避免数据泄漏）
        input_scaler = StandardScaler()
        train_input_data = X_train.reshape(-1, X_train.shape[-1])
        input_scaler.fit(train_input_data)
        joblib.dump(input_scaler, input_scaler_path)
        
        # 输出数据标准化器（仅使用训练集拟合，避免数据泄漏）
        output_scaler = StandardScaler()
        train_output_data = y_train.reshape(-1, y_train.shape[-1])
        output_scaler.fit(train_output_data)
        joblib.dump(output_scaler, output_scaler_path)
        
        print(f"    ✅ 标准化器已保存")
    
    if SKIP_DATA_FIGURES:
        print("\n⏭️  已跳过数据集可视化生成 (HGV_SKIP_DATA_FIGURES=1)")
        return

    # 4. 生成可视化
    print("\n🎨 生成控制参数分析图...")
    simulator.generate_control_parameters_analysis()
    
    print("🎨 生成3D轨迹投影图...")
    simulator.generate_3d_trajectory_plots()
    
    # 生成数据集多样性可视化（四张独立图片）
    print("\n🎨 生成数据集多样性可视化...")
    raw_trajectories_for_figures = load_raw_trajectories_as_dicts(raw_trajectories_path)
    simulator.generate_dataset_diversity_figures(raw_trajectories_for_figures, n_samples_per_type=3)

if __name__ == "__main__":
    main()
