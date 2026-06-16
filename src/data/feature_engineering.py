# -*- coding: utf-8 -*-
"""
特征工程模块：extract_manual_features_from_3d
从 3D 下颌运动轨迹中提取 16 维临床可解释特征（消融实验用）。
"""

import numpy as np


def extract_manual_features_from_3d(X_3d):
    """
    从 3D 下颌运动轨迹数据 (n, T, 3) 中提取 16 维临床可解释特征，
    用于消融实验"传统临床特征基线（w/o Bi-LSTM）"。

    坐标轴约定：
        axis-0 (x): 前后方向（anterior-posterior）
        axis-1 (y): 左右侧向偏斜（medial-lateral）
        axis-2 (z): 垂直开合方向（superior-inferior，张口度主轴）

    提取的 16 个特征均可对应 TMD 临床评估指标：
        F01  最大张口度          max(z)
        F02  开口幅度（极差）    max(z) − min(z)
        F03  最大侧向偏斜        max(|y|)
        F04  偏斜比              max(|y|) / (max(z)+ε)
        F05  前后位移幅度        max(|x|)
        F06  轨迹总路径长度      Σ‖Δpoint‖₂
        F07  轨迹线性度          直线距离 / 路径长度
        F08  平均运动速度        mean(‖Δpoint‖₂)
        F09  速度峰值            max(‖Δpoint‖₂)
        F10  速度标准差          std(‖Δpoint‖₂)
        F11  开口相速度峰值      max(|Δz|) in first-half
        F12  闭口相速度峰值      max(|Δz|) in second-half
        F13  开闭对称比          F11 / (F12 + ε)
        F14  平均曲率            mean(|Δθ|)
        F15  抖动度（Jerk）      mean(|Δ²z|)
        F16  侧向偏斜均值（开口相）mean(|y|) in first-half

    输出维度：(n_samples, 16)
    """
    n, T, _ = X_3d.shape
    eps = 1e-8
    features = []

    for i in range(n):
        traj = X_3d[i]           # (T, 3)
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

        half = T // 2
        z_open = z[:half]
        z_close = z[half:]
        y_open = y[:half]

        delta = np.diff(traj, axis=0)               # (T-1, 3)
        speed = np.linalg.norm(delta, axis=1)        # (T-1,)

        path_len = np.sum(speed) + eps
        straight_dist = np.linalg.norm(traj[-1] - traj[0]) + eps
        linearity = straight_dist / path_len

        if len(delta) > 1:
            norms = np.linalg.norm(delta, axis=1, keepdims=True) + eps
            unit = delta / norms
            cos_a = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1)
            mean_curv = np.mean(np.arccos(cos_a))
        else:
            mean_curv = 0.0

        jerk = np.mean(np.abs(np.diff(z, n=2))) if T > 2 else 0.0

        f01 = np.max(z)
        f02 = np.max(z) - np.min(z)
        f03 = np.max(np.abs(y))
        f04 = f03 / (f01 + eps)
        f05 = np.max(np.abs(x))
        f06 = path_len
        f07 = linearity
        f08 = np.mean(speed)
        f09 = np.max(speed)
        f10 = np.std(speed)
        f11 = np.max(np.abs(np.diff(z_open)))  if len(z_open) > 1 else 0.0
        f12 = np.max(np.abs(np.diff(z_close))) if len(z_close) > 1 else 0.0
        f13 = f11 / (f12 + eps)
        f14 = float(mean_curv)
        f15 = float(jerk)
        f16 = np.mean(np.abs(y_open))

        features.append([f01, f02, f03, f04, f05,
                          f06, f07, f08, f09, f10,
                          f11, f12, f13, f14, f15, f16])

    return np.array(features, dtype=np.float32)
