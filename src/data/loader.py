# -*- coding: utf-8 -*-
"""
数据加载模块：TrajectoryDataLoader + ConfidenceFilter
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


class TrajectoryDataLoader:
    def __init__(self, target_length=200, threshold=1.0):
        self.target_length = target_length
        self.threshold = threshold

    def load_dataset(self, normal_path, patient1_path, patient2_path):
        normal_data, normal_labels, normal_info = self._load_folder(normal_path, label=0)
        patient1_data, patient1_labels, patient1_info = self._load_folder(patient1_path, label=1)
        patient2_data, patient2_labels, patient2_info = self._load_folder(patient2_path, label=2)

        # 原始轨迹插值填充（不做截断）
        normal_padded = [self._interpolate_or_pad(traj) for traj in normal_data]
        patient1_padded = [self._interpolate_or_pad(traj) for traj in patient1_data]
        patient2_padded = [self._interpolate_or_pad(traj) for traj in patient2_data]
        X_raw = np.vstack((normal_padded, patient1_padded, patient2_padded))

        # 完整处理（截断平台期 + 插值）
        X_processed = self._process_all_data(normal_data, patient1_data, patient2_data)

        y = np.hstack((normal_labels, patient1_labels, patient2_labels))
        file_info = normal_info + patient1_info + patient2_info

        return X_raw, X_processed, y, file_info

    def _load_folder(self, folder_path, label):
        data, labels, file_info = [], [], []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.xlsx'):
                xls = pd.ExcelFile(os.path.join(folder_path, file_name))
                for sheet_index, sheet_name in enumerate(xls.sheet_names):
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=None).iloc[1:]
                    sequence = df.values
                    data.append(sequence)
                    labels.append(label)
                    file_info.append((file_name, sheet_index))
        return data, np.array(labels), file_info

    def _truncate_platform(self, trajectory, min_consecutive=2):
        n = len(trajectory)
        start = 0
        while start <= n - min_consecutive:
            if all(np.linalg.norm(trajectory[start + i]) < self.threshold
                   for i in range(min_consecutive)):
                start += 1
            else:
                break
        end = n
        while end >= min_consecutive:
            if all(np.max(np.abs(trajectory[end - i - 1])) < self.threshold
                   for i in range(min_consecutive)):
                end -= 1
            else:
                break
        return trajectory[start:end] if start < end else trajectory

    def _interpolate_or_pad(self, trajectory):
        current_length = len(trajectory)
        if current_length == self.target_length:
            return trajectory
        if current_length == 0:
            return np.zeros((self.target_length, 3))
        x_old = np.linspace(0, 1, current_length)
        x_new = np.linspace(0, 1, self.target_length)
        interpolated = np.zeros((self.target_length, trajectory.shape[1]))
        for dim in range(trajectory.shape[1]):
            if current_length > 1:
                f = interp1d(x_old, trajectory[:, dim], kind='linear',
                             fill_value="extrapolate")
                interpolated[:, dim] = f(x_new)
            else:
                interpolated[:, dim] = trajectory[0, dim]
        return interpolated

    def _process_trajectory(self, trajectory):
        truncated = self._truncate_platform(trajectory)
        return self._interpolate_or_pad(truncated)

    def _process_all_data(self, normal, p1, p2):
        norm_processed = [self._process_trajectory(traj) for traj in normal]
        p1_processed = [self._process_trajectory(traj) for traj in p1]
        p2_processed = [self._process_trajectory(traj) for traj in p2]
        return np.vstack((norm_processed, p1_processed, p2_processed))


class ConfidenceFilter:
    def __init__(self, model, threshold=0.5, visualize=True):
        self.model = clone(model)
        self.threshold = threshold
        self.visualize = False
        self.best_threshold = None
        self.original_acc = None
        self.filtered_acc = None

    def visualize_distribution(self, X, y):
        probs = self.model.fit(X, y).predict_proba(X)
        max_probs = np.max(probs, axis=1)
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=20, edgecolor='black', alpha=0.7)
        for t in [0.6, 0.7, 0.8]:
            plt.axvline(x=t, color='r', linestyle='--', linewidth=1.5,
                        label=f'Threshold {t}')
        plt.title("Confidence Score Distribution")
        plt.xlabel("Max Class Probability")
        plt.ylabel("Sample Count")
        plt.legend()
        plt.grid(True)
        plt.show()

    def optimize_threshold(self, X, y, cv=5):
        probs = self.model.fit(X, y).predict_proba(X)
        max_probs = np.max(probs, axis=1)
        thresholds = np.linspace(0.5, 0.95, 20)
        scores = []
        for tau in thresholds:
            mask = max_probs >= tau
            if np.unique(y[mask]).size < np.unique(y).size:
                continue
            score = cross_val_score(clone(self.model), X[mask], y[mask],
                                    cv=cv, scoring='accuracy').mean()
            scores.append(score)
        best_idx = np.argmax(scores)
        self.best_threshold = thresholds[best_idx]
        if self.visualize:
            plt.figure(figsize=(8, 5))
            plt.plot(thresholds[:len(scores)], scores, 'o-')
            plt.axvline(self.best_threshold, color='r', linestyle='--')
            plt.title(f"Optimal Threshold = {self.best_threshold:.2f}")
            plt.xlabel("Threshold")
            plt.ylabel("Cross-Val Accuracy")
            plt.grid(True)
            plt.show()
        return self.best_threshold

    def filter(self, X, y, threshold=None):
        threshold = threshold if threshold else self.threshold
        probs = self.model.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        mask = max_probs >= threshold
        return X[mask], y[mask], np.where(~mask)[0], mask

    def filter_pipeline(self, X_train, y_train, X_test, y_test, file_info):
        print("\n🚦 开始置信度筛选流程...")
        self.original_acc = self.model.fit(X_train, y_train).score(X_test, y_test)
        print(f"原始准确率：{self.original_acc:.4f}")
        if self.visualize:
            self.visualize_distribution(X_train, y_train)

        best_thresh = self.optimize_threshold(X_train, y_train)
        effective_thresh = min(self.threshold, best_thresh)
        print(f"使用阈值：{effective_thresh:.2f}（最优建议：{best_thresh:.2f}）")

        X_filtered, y_filtered, removed_indices, mask = self.filter(
            X_train, y_train, threshold=effective_thresh)

        removed_samples = {'low_confidence': []}
        file_info_len = len(file_info) if file_info is not None else 0
        for i in removed_indices:
            if i < file_info_len:
                removed_samples['low_confidence'].append(file_info[i])
            else:
                removed_samples['low_confidence'].append(
                    (f"SMOTE_Generated_Sample_{i}", -1))

        print(f"过滤后样本数：{len(X_filtered)} / {len(X_train)}")
        print(f"被移除样本数量：{len(removed_indices)}")

        self.filtered_acc = clone(self.model).fit(X_filtered, y_filtered).score(
            X_test, y_test)
        print(f"过滤后准确率：{self.filtered_acc:.4f}")

        print("\n被过滤的样本信息:")
        for filter_type, samples in removed_samples.items():
            print(f"{filter_type} 移除了 {len(samples)} 个样本:")
            for file_name, sheet_idx in samples:
                if sheet_idx == -1:
                    print(f" - {file_name} (SMOTE生成的合成样本)")
                else:
                    print(f" - 文件: {file_name}, 工作表: {sheet_idx}")

        print("训练集在经过数据筛选后大小：")
        print(X_filtered.shape)

        if self.visualize:
            self._evaluate_acc_change()

        return X_filtered, y_filtered, removed_samples, mask

    def _evaluate_acc_change(self):
        diff = (self.filtered_acc - self.original_acc) * 100 / self.original_acc
        print(f"准确率提升：{diff:.2f}%")
        plt.bar(['Original', 'Filtered'], [self.original_acc, self.filtered_acc])
        plt.ylabel("Accuracy")
        plt.title("Effect of Confidence Filtering")
        plt.grid(True)
        plt.show()
