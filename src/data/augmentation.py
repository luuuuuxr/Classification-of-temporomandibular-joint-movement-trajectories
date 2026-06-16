# -*- coding: utf-8 -*-
"""
数据增强模块：TrajectoryAugmenter + apply_smote + data_split_and_augment
             + augment_data_pipeline + filter_data_pipeline
"""

import random
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from imblearn.over_sampling import SMOTE
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split


class TrajectoryAugmenter:
    def __init__(self, config=None, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

        self.config = config or {
            'max_time_warp': 0.3,
            'noise_scale': 0.15,
            'keypoint_shift': 0.2,
            'smoothness': 1.5
        }
        self.methods_config = {
            'dtw': {'n_samples': 1},
            'time_warp': {},
            'physics': {},
        }

    # ── 参考轨迹生成 ──────────────────────────────────────────

    def _generate_reference(self, trajectory):
        if len(trajectory) < 3:
            return trajectory.copy()
        rng = np.random.RandomState(self.random_state)
        method = rng.choice(
            ['time_warp', 'noisy_template', 'keypoint_displacement'],
            p=[0.4, 0.4, 0.2]
        )
        if method == 'time_warp':
            return self._time_warp_reference(trajectory)
        elif method == 'noisy_template':
            return self._noisy_template(trajectory)
        else:
            return self._keypoint_displacement(trajectory)

    def _time_warp_reference(self, trajectory):
        n_points = len(trajectory)
        rng = np.random.RandomState(self.random_state)
        warp_factor = 1 + rng.uniform(
            -self.config['max_time_warp'], self.config['max_time_warp'])
        original_time = np.linspace(0, 1, n_points)
        new_time = np.linspace(0, 1, int(n_points * warp_factor))
        ref_traj = np.zeros((len(new_time), trajectory.shape[1]))
        for dim in range(trajectory.shape[1]):
            interp_fn = interpolate.interp1d(
                original_time, trajectory[:, dim], kind='quadratic')
            ref_traj[:, dim] = interp_fn(new_time)
        return ref_traj

    def _noisy_template(self, trajectory):
        window_size = max(3, int(len(trajectory) * 0.1))
        smoothed = np.array([
            np.mean(trajectory[max(0, i - window_size):i + window_size], axis=0)
            for i in range(len(trajectory))
        ])
        rng = np.random.RandomState(self.random_state)
        noise = rng.normal(scale=self.config['noise_scale'], size=trajectory.shape)
        noisy_ref = smoothed + noise
        return gaussian_filter1d(noisy_ref, sigma=self.config['smoothness'], axis=0)

    def _keypoint_displacement(self, trajectory):
        key_idx = np.random.choice(
            len(trajectory), size=int(len(trajectory) * 0.3), replace=False)
        displacement = np.random.uniform(
            -self.config['keypoint_shift'], self.config['keypoint_shift'],
            size=(len(key_idx), trajectory.shape[1])
        )
        ref_traj = trajectory.copy()
        for i, idx in enumerate(key_idx):
            radius = np.random.randint(1, 3)
            start = max(0, idx - radius)
            end = min(len(trajectory), idx + radius + 1)
            weights = np.linspace(0, 1, end - start)[:, np.newaxis]
            ref_traj[start:end] += displacement[i] * weights
        return gaussian_filter1d(ref_traj, sigma=1.0, axis=0)

    # ── 增强方法 ──────────────────────────────────────────────

    def dtw_augment(self, trajectory, n_samples=1):
        augmented = []
        for _ in range(n_samples):
            ref = self._generate_reference(trajectory)
            _, path = fastdtw(trajectory, ref, dist=self._cosine_distance)
            new_traj = ref[np.array([p[1] for p in path])]
            augmented.append(new_traj)
        return augmented

    def noise_augment(self, trajectory, n_samples=1):
        return [trajectory + np.random.normal(scale=0.1, size=trajectory.shape)
                for _ in range(n_samples)]

    def time_warp(self, trajectory, n_anchor=3, max_shift=20):
        anchors = np.sort(np.random.choice(len(trajectory), n_anchor, replace=False))
        warped = []
        for i in range(len(anchors) - 1):
            start, end = anchors[i], anchors[i + 1]
            segment = trajectory[start:end]
            if len(segment) < 2:
                continue
            scale = np.random.uniform(0.8, 1.2)
            new_length = max(2, int(len(segment) * scale))
            x = np.linspace(0, 1, len(segment))
            x_new = np.linspace(0, 1, new_length)
            interpolated_segment = np.stack([
                np.interp(x_new, x, segment[:, dim])
                for dim in range(segment.shape[1])
            ], axis=-1)
            warped.append(interpolated_segment)
        if not warped:
            return [trajectory.copy()]
        return [np.vstack(warped)]

    def physics_augment(self, trajectory, noise_level=0.1):
        velocity = np.gradient(trajectory, axis=0)
        acceleration = np.gradient(velocity, axis=0)
        noise_v = np.random.normal(0, noise_level, velocity.shape)
        noise_a = np.random.normal(0, noise_level, acceleration.shape)
        new_traj = np.zeros_like(trajectory)
        new_traj[0] = trajectory[0]
        for t in range(1, len(trajectory)):
            new_traj[t] = (new_traj[t - 1]
                           + (velocity[t] + noise_v[t])
                           + 0.5 * (acceleration[t] + noise_a[t]))
        return [new_traj]

    def trajectory_shuffle(self, trajectory):
        speeds = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        high_speed_indices = np.where(speeds > np.percentile(speeds, 70))[0]
        if len(high_speed_indices) < 2:
            return [trajectory]
        split_points = (np.sort(
            np.random.choice(high_speed_indices, size=2, replace=False)) + 1)
        segments = np.split(trajectory, split_points)
        np.random.shuffle(segments)
        return [np.concatenate(segments, axis=0)]

    def _cosine_distance(self, a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    # ── 批量生成 ──────────────────────────────────────────────

    def _generate_for_class(self, data, needed, label=None, track=False, start_index=0):
        samples = []
        track_info = []
        count = 0
        label_int = int(label) if label is not None else None
        seed_value = (self.random_state + label_int
                      if label_int is not None else self.random_state)
        rng = np.random.RandomState(seed_value)
        random_rng = random.Random(seed_value)

        while len(samples) < needed:
            base_index = rng.choice(len(data))
            base_sample = data[base_index]
            method_fn = random_rng.choice([
                ('dtw', partial(self.dtw_augment, **self.methods_config['dtw'])),
                ('time_warp', partial(self.time_warp, **self.methods_config['time_warp'])),
                ('physics', partial(self.physics_augment, **self.methods_config['physics'])),
                ('shuffle', self.trajectory_shuffle)
            ])
            method_name, method = method_fn
            try:
                new_samples = method(base_sample)
                valid_samples = self._filter_valid_trajectories(new_samples)
                for sample in valid_samples:
                    if len(samples) >= needed:
                        break
                    samples.append(sample)
                    if track:
                        track_info.append({
                            'index': start_index + count,
                            'source_index': base_index,
                            'label': label,
                            'method': method_name
                        })
                    count += 1
            except Exception as e:
                print(f"[Error] {method_name} generation failed: {e}")
                continue

        print(f"[DEBUG] 类别 {label} 实际生成样本数：{len(samples)}")
        return samples, track_info

    def split_augmented_by_label(self, X_aug, y_aug):
        augmented_data = []
        for label in np.unique(y_aug):
            indices = np.where(y_aug == label)[0]
            augmented_data.append((X_aug[indices], label))
        return augmented_data

    def _combine_augmented(self, X, y, augmented_data, target_length=200):
        X_list = [X]
        y_list = [y]

        def standardize_length(trajectory, tgt=200):
            current_len = trajectory.shape[0]
            if current_len == tgt:
                return trajectory
            x_old = np.linspace(0, 1, current_len)
            x_new = np.linspace(0, 1, tgt)
            return np.stack([
                np.interp(x_new, x_old, trajectory[:, dim])
                for dim in range(trajectory.shape[1])
            ], axis=-1)

        for samples, label in augmented_data:
            standardized = []
            for traj in samples:
                if traj.ndim == 2 and traj.shape[1] == 3:
                    standardized.append(standardize_length(traj, target_length))
            if standardized:
                arr = np.array(standardized)
                X_list.append(arr)
                y_list.append(np.full(len(arr), label))
            else:
                print(f"[Warning] 所有增强样本 shape 无效或空，类别 {label} 被跳过")

        combined_X = np.concatenate(X_list, axis=0)
        combined_y = np.concatenate(y_list, axis=0)
        shuffle_idx = np.random.permutation(len(combined_X))
        return combined_X[shuffle_idx], combined_y[shuffle_idx]

    def apply_augmentation(self, X, y, target_counts=None, track=False):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        augmented_data = []
        all_track_info = []
        start_idx = 0
        print("类别索引：", np.unique(y))
        print("目标数量：", target_counts)

        for class_idx in np.unique(y):
            class_data = X[y == class_idx]
            needed = target_counts[class_idx] - len(class_data)
            if needed < 0:
                continue
            samples, track_info = self._generate_for_class(
                data=class_data, needed=needed, label=class_idx,
                track=track, start_index=start_idx
            )
            start_idx += len(samples)
            augmented_data.append((samples, class_idx))
            all_track_info.extend(track_info)

        combined_X, combined_y = self._combine_augmented(X, y, augmented_data)
        return combined_X, combined_y, all_track_info

    def _filter_valid_trajectories(self, trajectories):
        return [
            traj for traj in trajectories
            if (traj is not None and isinstance(traj, np.ndarray)
                and traj.ndim == 2 and traj.shape[1] == 3)
        ]

    @staticmethod
    def save_augmentation_log(track_info, save_path='augmentation_log.xlsx'):
        df = pd.DataFrame(track_info)
        df.to_excel(save_path, index=False)
        print(f"增强追踪日志已保存至: {save_path}")

    def visualize_augmentation_pair(self, original, augmented, title='Augmentation Comparison'):
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(original[:, 0], original[:, 1], original[:, 2], label='Original', linewidth=2)
        ax.plot(augmented[:, 0], augmented[:, 1], augmented[:, 2], label='Augmented', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.grid(True)
        plt.show()


# ──────────────────────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────────────────────

def apply_smote(X, y, expand_factor=1, target_counts=None):
    """使用 SMOTE 进行过采样，并按照指定的类别目标数量进行重采样。"""
    np.random.seed(42)
    class_counts = Counter(y)
    print("原始数据类别分布:", class_counts)

    if target_counts is None:
        default_targets = [127, 50, 50]
        sorted_labels = sorted(class_counts.keys())
        target_counts = {}
        for idx, label in enumerate(sorted_labels):
            target_counts[label] = (default_targets[idx]
                                    if idx < len(default_targets)
                                    else class_counts[label])
    else:
        target_counts = {int(k): int(v) for k, v in target_counts.items()}

    adjusted_strategy = {}
    for label, orig_count in class_counts.items():
        desired = target_counts.get(label, orig_count)
        if desired < orig_count:
            print(f"⚠️ 目标数量 {desired} 小于原始数量 {orig_count}，已自动调整。")
            desired = orig_count
        adjusted_strategy[label] = desired

    print("SMOTE 目标采样策略:", adjusted_strategy)
    smote = SMOTE(sampling_strategy=adjusted_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X.reshape(X.shape[0], -1), y)
    print("过采样后类别分布:", Counter(y_resampled))
    X_resampled = X_resampled.reshape(-1, X.shape[1], X.shape[2])

    if expand_factor > 1.0:
        int_expand = int(np.floor(expand_factor))
        frac_expand = expand_factor - int_expand
        for i in range(int_expand - 1):
            rng = np.random.RandomState(42 + i)
            random_indices = rng.choice(X_resampled.shape[0],
                                        X_resampled.shape[0], replace=True)
            X_resampled = np.concatenate(
                (X_resampled, X_resampled[random_indices]), axis=0)
            y_resampled = np.concatenate(
                (y_resampled, y_resampled[random_indices]), axis=0)
        if frac_expand > 0:
            sample_size = int(X_resampled.shape[0] * frac_expand)
            rng = np.random.RandomState(42 + int_expand)
            random_indices = rng.choice(X_resampled.shape[0], sample_size, replace=False)
            X_resampled = np.concatenate(
                (X_resampled, X_resampled[random_indices]), axis=0)
            y_resampled = np.concatenate(
                (y_resampled, y_resampled[random_indices]), axis=0)

    return X_resampled, y_resampled


def data_split_and_augment(X, y):
    print("数据初始分布：", Counter(y))
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, stratify=y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=0.8, random_state=42)
    print("训练集数据分布：", Counter(y_train))
    print("测试集数据分布：", Counter(y_test))
    print("验证集数据分布：", Counter(y_val))
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    print("训练集初始大小：", X_train.shape)
    print("训练集在经过SMOTE后大小：", X_train_resampled.shape)
    print("测试集大小：", X_test.shape)
    return X_train, y_train, X_train_resampled, y_train_resampled, X_val, X_test, y_val, y_test


def augment_data_pipeline(X_train, y_train, augmenter, target_counts,
                           scaler=None, save_log_path="augmentation_log.xlsx"):
    """
    对训练数据进行增强，输出合并后的数据和增强追踪日志。
    """
    print("\n🔁 正在进行数据增强...")
    X_combined, y_combined, track_info = augmenter.apply_augmentation(
        X_train, y_train, target_counts, track=True)

    X_augmented_only = X_combined[len(X_train):]
    y_augmented_only = y_combined[len(y_train):]
    aug_file_info = []
    for (samples, label) in augmenter.split_augmented_by_label(
            X_augmented_only, y_augmented_only):
        for i in range(len(samples)):
            aug_file_info.append((f"Augmented_Class{label}", i))

    if save_log_path is not None:
        TrajectoryAugmenter.save_augmentation_log(track_info, save_path=save_log_path)
    print(f"✅ 增强完成: 原始数量={len(X_train)}, 增强后总数量={len(X_combined)}")
    return X_combined, y_combined, track_info


def filter_data_pipeline(X, y, model, file_info, threshold,
                          X_test=None, y_test=None, visualize=True):
    """
    封装 ConfidenceFilter 的统一调用接口。
    """
    from src.data.loader import ConfidenceFilter
    print("\n🧹 [Pipeline] 启动置信度筛选...")
    confidence_filter = ConfidenceFilter(model=model, threshold=threshold,
                                         visualize=visualize)
    X_filtered, y_filtered, removed_samples, mask = confidence_filter.filter_pipeline(
        X_train=X, y_train=y, X_test=X_test, y_test=y_test, file_info=file_info)
    return X_filtered, y_filtered, removed_samples, mask
