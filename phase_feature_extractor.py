"""
基于原始轨迹的阶段特征提取模块
用于可解释性分析，确保临床意义
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from trajectory_build import split_into_phases

def extract_phase_statistical_features(phase_data):
    """
    提取单个阶段的统计特征
    
    参数:
        phase_data: 单个阶段的轨迹数据 (n_points, 3)
    
    返回:
        features: 特征向量
    """
    if len(phase_data) == 0:
        return np.zeros(20)  # 返回零特征
    
    features = []
    
    # 1. 基本统计特征
    for coord in range(3):  # x, y, z坐标
        coord_data = phase_data[:, coord]
        
        # 位置特征
        features.extend([
            np.mean(coord_data),      # 均值
            np.std(coord_data),       # 标准差
            np.var(coord_data),       # 方差
            np.median(coord_data),    # 中位数
            np.ptp(coord_data),       # 极差
            np.percentile(coord_data, 25),  # 25分位数
            np.percentile(coord_data, 75),  # 75分位数
        ])
    
    # 2. 运动特征
    if len(phase_data) > 1:
        # 计算速度
        velocity = np.diff(phase_data, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        
        features.extend([
            np.mean(speed),           # 平均速度
            np.std(speed),            # 速度标准差
            np.max(speed),            # 最大速度
            np.min(speed),            # 最小速度
        ])
        
        # 计算加速度
        if len(velocity) > 1:
            acceleration = np.diff(velocity, axis=0)
            accel_magnitude = np.linalg.norm(acceleration, axis=1)
            features.extend([
                np.mean(accel_magnitude),  # 平均加速度
                np.std(accel_magnitude),   # 加速度标准差
            ])
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 3. 轨迹形状特征
    if len(phase_data) > 2:
        # 轨迹长度
        trajectory_length = np.sum(np.linalg.norm(np.diff(phase_data, axis=0), axis=1))
        features.append(trajectory_length)
        
        # 直线距离
        straight_distance = np.linalg.norm(phase_data[-1] - phase_data[0])
        features.append(straight_distance)
        
        # 轨迹复杂度（直线距离/轨迹长度）
        complexity = straight_distance / (trajectory_length + 1e-8)
        features.append(complexity)
    else:
        features.extend([0.0, 0.0, 0.0])
    
    return np.array(features)

def extract_phase_frequency_features(phase_data):
    """
    提取单个阶段的频域特征
    
    参数:
        phase_data: 单个阶段的轨迹数据 (n_points, 3)
    
    返回:
        features: 频域特征向量
    """
    if len(phase_data) < 4:
        return np.zeros(15)  # 返回零特征
    
    features = []
    
    for coord in range(3):  # x, y, z坐标
        coord_data = phase_data[:, coord]
        
        # FFT变换
        fft_data = fft(coord_data)
        freqs = fftfreq(len(coord_data))
        
        # 只取正频率部分
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_data[:len(fft_data)//2])
        
        if len(positive_fft) > 0:
            # 频域特征
            features.extend([
                np.max(positive_fft),                    # 最大幅值
                np.mean(positive_fft),                   # 平均幅值
                np.std(positive_fft),                    # 幅值标准差
                positive_freqs[np.argmax(positive_fft)], # 主频率
                np.sum(positive_fft),                    # 总能量
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    return np.array(features)

def extract_phase_features_all_trajectories(trajectories, num_phases=6):
    """
    为所有轨迹提取阶段特征
    
    参数:
        trajectories: 轨迹数据 (n_samples, time_steps, 3)
        num_phases: 阶段数量
    
    返回:
        phase_features: 阶段特征 (n_samples, num_phases, feature_dim)
        phase_names: 阶段名称列表
    """
    all_phase_features = []
    
    for trajectory in trajectories:
        # 分割轨迹为阶段
        phases = split_into_phases(trajectory, num_phases)
        
        trajectory_phase_features = []
        for phase in phases:
            # 提取统计特征
            stat_features = extract_phase_statistical_features(phase)
            # 提取频域特征
            freq_features = extract_phase_frequency_features(phase)
            # 合并特征
            phase_features = np.concatenate([stat_features, freq_features])
            trajectory_phase_features.append(phase_features)
        
        all_phase_features.append(trajectory_phase_features)
    
    return np.array(all_phase_features)

def create_phase_feature_matrix(phase_features):
    """
    将阶段特征转换为适合机器学习的形式
    
    参数:
        phase_features: 阶段特征 (n_samples, num_phases, feature_dim)
    
    返回:
        X_phase: 展平的特征矩阵 (n_samples, num_phases * feature_dim)
        phase_info: 阶段信息字典
    """
    n_samples, num_phases, feature_dim = phase_features.shape
    
    # 展平特征
    X_phase = phase_features.reshape(n_samples, -1)
    
    # 创建阶段信息
    phase_info = {
        
num_phases: num_phases,
        feature_dim: feature_dim,
        phase_names: [fPhase_
i+1
 for i in range(num_phases)],
        
feature_names: []
    }
    
    # 生成特征名称
    stat_feature_names = [
        mean_x, std_x, var_x, median_x, ptp_x, q25_x, q75_x,
        mean_y, std_y, var_y, median_y, ptp_y, q25_y, q75_y, 
        mean_z, std_z, var_z, median_z, ptp_z, q25_z, q75_z,
        mean_speed, std_speed, max_speed, min_speed,
        mean_accel, std_accel, traj_length, straight_dist, complexity
    ]
    
    freq_feature_names = [
        max_amp_x, mean_amp_x, std_amp_x, main_freq_x, total_energy_x,
        max_amp_y, mean_amp_y, std_amp_y, main_freq_y, total_energy_y,
        max_amp_z, mean_amp_z, std_amp_z, main_freq_z, total_energy_z
    ]
    
    for phase_idx in range(num_phases):
        for stat_name in stat_feature_names:
            phase_info[feature_names].append(fPhase_
phase_idx+1
_
stat_name
)
        for freq_name in freq_feature_names:
            phase_info[
feature_names].append(fPhase_
phase_idx+1
_
freq_name
)
    
    return X_phase, phase_info
