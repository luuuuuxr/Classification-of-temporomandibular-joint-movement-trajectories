import os
import random
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
from fastdtw import fastdtw
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE
from keras import Model, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Attention, Bidirectional, Conv1D, MaxPooling1D, \
    Concatenate, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.regularizers import l2
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.models.advanced_classifiers import test_advanced_classifiers, AdvancedClassifierTrainer

# 新增四种先进分类器的导入
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from src.visualization.model_explainability import plot_phase_shap_importance, plot_phase_aggregated_feature_importance
from src.visualization.improved_explainability import comprehensive_phase_analysis
from src.utils.trajectory_build import split_into_phases, plot_phases_3d
from src.models.advanced_attention import AdvancedAttentionMechanisms
from src.models.enhanced_lstm_model import EnhancedLSTMFeatureExtractor, create_enhanced_lstm_model

# 设置全局随机种子
tf.random.set_seed(42)


# ========== 四种新增先进分类器实现 ==========

class LightGBMClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError('LightGBM not available')
        X_scaled = self.scaler.fit_transform(X)
        train_data = lgb.Dataset(X_scaled, label=y)
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        self.model = lgb.train(params, train_data, 100)
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        pred_proba = self.model.predict(X_scaled, num_iteration=self.model.best_iteration)
        return np.argmax(pred_proba, axis=1)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, num_iteration=self.model.best_iteration)


class CatBoostClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        if not CATBOOST_AVAILABLE:
            raise ImportError('CatBoost not available')
        X_scaled = self.scaler.fit_transform(X)
        self.model = cb.CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function='MultiClass',
            random_seed=42
        )
        self.model.fit(X_scaled, y)
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # CatBoost 返回二维数组，需要压缩为一维
        return predictions.flatten() if predictions.ndim > 1 else predictions
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class PrototypicalNetwork:
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        self.prototypes = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        
    def _compute_prototypes(self, embeddings, labels):
        unique_labels = torch.unique(labels)
        prototypes = []
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                prototype = embeddings[mask].mean(dim=0)
                prototypes.append(prototype)
        return torch.stack(prototypes) if prototypes else torch.empty(0, embeddings.size(1))
    
    def _prototypical_loss(self, embeddings, labels, prototypes):
        if len(prototypes) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        distances = torch.cdist(embeddings, prototypes)
        logits = -distances
        return F.cross_entropy(logits, labels)
        
    def fit(self, X, y, epochs=200, batch_size=32, lr=0.001, patience=20):
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._build_encoder().to(self.device)
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, 
        )
        
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train_tensor), batch_size):
                batch_end = min(i + batch_size, len(X_train_tensor))
                batch_x = X_train_tensor[i:batch_end]
                batch_y = y_train_tensor[i:batch_end]
                
                optimizer.zero_grad()
                embeddings = self.model(batch_x)
                prototypes = self._compute_prototypes(embeddings, batch_y)
                loss = self._prototypical_loss(embeddings, batch_y, prototypes)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            self.model.eval()
            with torch.no_grad():
                val_embeddings = self.model(X_val_tensor)
                val_prototypes = self._compute_prototypes(val_embeddings, y_val_tensor)
                val_loss = self._prototypical_loss(val_embeddings, y_val_tensor, val_prototypes)
            self.model.train()
            
            avg_train_loss = total_loss / num_batches
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.model.eval()
        with torch.no_grad():
            all_embeddings = self.model(torch.FloatTensor(X_scaled).to(self.device))
            all_labels = torch.LongTensor(y).to(self.device)
            self.prototypes = self._compute_prototypes(all_embeddings, all_labels)
        
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(X_tensor)
            if len(self.prototypes) > 0:
                distances = torch.cdist(embeddings, self.prototypes)
                predictions = torch.argmin(distances, dim=1)
            else:
                predictions = torch.randint(0, 3, (len(X),), device=self.device)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(X_tensor)
            if len(self.prototypes) > 0:
                distances = torch.cdist(embeddings, self.prototypes)
                # 将距离转换为概率（使用softmax）
                logits = -distances  # 负距离作为logits
                probabilities = torch.softmax(logits, dim=1)
            else:
                # 如果没有原型，返回均匀分布
                probabilities = torch.ones(len(X), 3, device=self.device) / 3
        
        return probabilities.cpu().numpy()
        
    def set_training_data(self, X, y):
        self.X_train = X
        self.y_train = y

# ========== 新分类器实现结束 ==========

# ========== 新分类器实现结束 ==========


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

        '''plot_3d_trajectory(X, y, file_info, base_save_path='3D_Trajectories_original')
        plot_3d_trajectory(X_processed, y, file_info, base_save_path='3D_Trajectories_processed')
        plot_2d_trajectory(X, y, file_info, "2D_Plots_original")
        plot_2d_trajectory(X_processed, y, file_info, "2D_Plots_processed")'''

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
            if all(np.linalg.norm(trajectory[start + i]) < self.threshold for i in range(min_consecutive)):
                start += 1
            else:
                break
        end = n
        while end >= min_consecutive:
            if all(np.max(np.abs(trajectory[end - i - 1])) < self.threshold for i in range(min_consecutive)):
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
                f = interp1d(x_old, trajectory[:, dim], kind='linear', fill_value="extrapolate")
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
        """
        初始化筛选器

        :param model: 支持 predict_proba 的模型
        :param threshold: 用户设置的最低置信度阈值
        :param visualize: 是否启用可视化
        """
        self.model = clone(model)
        self.threshold = threshold
        self.visualize = False
        self.best_threshold = None
        self.original_acc = None
        self.filtered_acc = None

    def visualize_distribution(self, X, y):
        """绘制置信度直方图"""
        probs = self.model.fit(X, y).predict_proba(X)
        max_probs = np.max(probs, axis=1)

        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=20, edgecolor='black', alpha=0.7)

        for t in [0.6, 0.7, 0.8]:
            plt.axvline(x=t, color='r', linestyle='--', linewidth=1.5, label=f'Threshold {t}')
        plt.title("Confidence Score Distribution")
        plt.xlabel("Max Class Probability")
        plt.ylabel("Sample Count")
        plt.legend()
        plt.grid(True)
        plt.show()

    def optimize_threshold(self, X, y, cv=5):
        """自动选择最优阈值"""
        probs = self.model.fit(X, y).predict_proba(X)
        max_probs = np.max(probs, axis=1)

        thresholds = np.linspace(0.5, 0.95, 20)
        scores = []

        for tau in thresholds:
            mask = max_probs >= tau
            if np.unique(y[mask]).size < np.unique(y).size:
                continue
            score = cross_val_score(clone(self.model), X[mask], y[mask], cv=cv, scoring='accuracy').mean()
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
        """根据指定阈值筛选样本"""
        threshold = threshold if threshold else self.threshold
        probs = self.model.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        mask = max_probs >= threshold

        return X[mask], y[mask], np.where(~mask)[0], mask

    def filter_pipeline(self, X_train, y_train, X_test, y_test, file_info):
        """执行完整的置信度筛选流程，返回过滤数据及被删样本信息"""
        print("\n🚦 开始置信度筛选流程...")
        self.original_acc = self.model.fit(X_train, y_train).score(X_test, y_test)
        print(f"原始准确率：{self.original_acc:.4f}")

        if self.visualize:
            self.visualize_distribution(X_train, y_train)

        best_thresh = self.optimize_threshold(X_train, y_train)
        effective_thresh = min(self.threshold, best_thresh)
        print(f"使用阈值：{effective_thresh:.2f}（最优建议：{best_thresh:.2f}）")

        X_filtered, y_filtered, removed_indices, mask = self.filter(X_train, y_train, threshold=effective_thresh)
        removed_samples = {'low_confidence': [file_info[i] for i in removed_indices]}

        print(f"过滤后样本数：{len(X_filtered)} / {len(X_train)}")
        print(f"被移除样本数量：{len(removed_indices)}")

        # 新模型评估
        self.filtered_acc = clone(self.model).fit(X_filtered, y_filtered).score(X_test, y_test)
        print(f"过滤后准确率：{self.filtered_acc:.4f}")

        # 打印被过滤的样本信息
        print("\n被过滤的样本信息:")
        for filter_type, samples in removed_samples.items():
            print(f"{filter_type} 移除了 {len(samples)} 个样本:")
            for file_name, sheet_idx in samples:
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


class TrajectoryAugmenter:
    def __init__(self, config=None):
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

    def _generate_reference(self, trajectory):
        if len(trajectory) < 3:
            return trajectory.copy()

        method = np.random.choice(['time_warp', 'noisy_template', 'keypoint_displacement'], p=[0.4, 0.4, 0.2])
        if method == 'time_warp':
            return self._time_warp_reference(trajectory)
        elif method == 'noisy_template':
            return self._noisy_template(trajectory)
        else:
            return self._keypoint_displacement(trajectory)

    def _time_warp_reference(self, trajectory):
        n_points = len(trajectory)
        warp_factor = 1 + np.random.uniform(-self.config['max_time_warp'], self.config['max_time_warp'])
        original_time = np.linspace(0, 1, n_points)
        new_time = np.linspace(0, 1, int(n_points * warp_factor))

        ref_traj = np.zeros((len(new_time), trajectory.shape[1]))
        for dim in range(trajectory.shape[1]):
            interp_fn = interpolate.interp1d(original_time, trajectory[:, dim], kind='quadratic')
            ref_traj[:, dim] = interp_fn(new_time)

        return ref_traj

    def _noisy_template(self, trajectory):
        window_size = max(3, int(len(trajectory) * 0.1))
        smoothed = np.array([np.mean(trajectory[max(0, i - window_size):i + window_size], axis=0)
                             for i in range(len(trajectory))])

        noise = np.random.normal(scale=self.config['noise_scale'], size=trajectory.shape)
        noisy_ref = smoothed + noise

        return gaussian_filter1d(noisy_ref, sigma=self.config['smoothness'], axis=0)

    def _keypoint_displacement(self, trajectory):
        key_idx = np.random.choice(len(trajectory), size=int(len(trajectory) * 0.3), replace=False)
        displacement = np.random.uniform(-self.config['keypoint_shift'], self.config['keypoint_shift'],
                                         size=(len(key_idx), trajectory.shape[1]))

        ref_traj = trajectory.copy()
        for i, idx in enumerate(key_idx):
            radius = np.random.randint(1, 3)
            start = max(0, idx - radius)
            end = min(len(trajectory), idx + radius + 1)
            weights = np.linspace(0, 1, end - start)[:, np.newaxis]
            ref_traj[start:end] += displacement[i] * weights

        return gaussian_filter1d(ref_traj, sigma=1.0, axis=0)

    def dtw_augment(self, trajectory, n_samples=1):
        augmented = []
        for _ in range(n_samples):
            ref = self._generate_reference(trajectory)
            _, path = fastdtw(trajectory, ref, dist=self._cosine_distance)
            new_traj = ref[np.array([p[1] for p in path])]
            augmented.append(new_traj)
        return augmented

    def noise_augment(self, trajectory, n_samples=1):
        return [trajectory + np.random.normal(scale=0.1, size=trajectory.shape) for _ in range(n_samples)]

    def time_warp(self, trajectory, n_anchor=3, max_shift=20):
        anchors = np.sort(np.random.choice(len(trajectory), n_anchor, replace=False))
        warped = []

        for i in range(len(anchors) - 1):
            start, end = anchors[i], anchors[i + 1]
            segment = trajectory[start:end]

            if len(segment) < 2:
                continue  # 跳过无法插值的段

            scale = np.random.uniform(0.8, 1.2)
            new_length = max(2, int(len(segment) * scale))  # 确保不低于2
            x = np.linspace(0, 1, len(segment))
            x_new = np.linspace(0, 1, new_length)

            # 对每个维度做线性插值
            interpolated_segment = np.stack([
                np.interp(x_new, x, segment[:, dim]) for dim in range(segment.shape[1])
            ], axis=-1)

            warped.append(interpolated_segment)

        if not warped:
            return [trajectory.copy()]  # 如果所有段都跳过，返回原始

        return [np.vstack(warped)]

    def physics_augment(self, trajectory, noise_level=0.1):
        velocity = np.gradient(trajectory, axis=0)
        acceleration = np.gradient(velocity, axis=0)

        noise_v = np.random.normal(0, noise_level, velocity.shape)
        noise_a = np.random.normal(0, noise_level, acceleration.shape)

        new_traj = np.zeros_like(trajectory)
        new_traj[0] = trajectory[0]

        for t in range(1, len(trajectory)):
            new_traj[t] = new_traj[t - 1] + (velocity[t] + noise_v[t]) + 0.5 * (acceleration[t] + noise_a[t])

        return [new_traj]

    def trajectory_shuffle(self, trajectory):
        speeds = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        high_speed_indices = np.where(speeds > np.percentile(speeds, 70))[0]

        if len(high_speed_indices) < 2:
            return [trajectory]

        split_points = np.sort(np.random.choice(high_speed_indices, size=2, replace=False)) + 1
        segments = np.split(trajectory, split_points)
        np.random.shuffle(segments)
        return [np.concatenate(segments, axis=0)]

    def _cosine_distance(self, a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def _generate_for_class(self, data, needed, label=None, track=False, start_index=0):
        samples = []
        track_info = []

        count = 0
        while len(samples) < needed:
            base_index = np.random.choice(len(data))
            base_sample = data[base_index]

            method_fn = random.choice([
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

    # 注意：确保 X_augmented 和 y_augmented 是增强部分，不包含原始数据
    def split_augmented_by_label(self, X_aug, y_aug):
        augmented_data = []
        labels = np.unique(y_aug)
        for label in labels:
            indices = np.where(y_aug == label)[0]
            class_samples = X_aug[indices]
            augmented_data.append((class_samples, label))
        return augmented_data

    def _combine_augmented(self, X, y, augmented_data, target_length=200):
        X_list = [X]
        y_list = [y]

        def standardize_length(trajectory, target_length=200):
            """插值或填充所有轨迹为固定长度"""
            current_len = trajectory.shape[0]
            if current_len == target_length:
                return trajectory
            x_old = np.linspace(0, 1, current_len)
            x_new = np.linspace(0, 1, target_length)
            new_traj = np.stack([
                np.interp(x_new, x_old, trajectory[:, dim]) for dim in range(trajectory.shape[1])
            ], axis=-1)
            return new_traj

        for samples, label in augmented_data:
            standardized_samples = []
            for traj in samples:
                if traj.ndim == 2 and traj.shape[1] == 3:
                    fixed_traj = standardize_length(traj, target_length)
                    standardized_samples.append(fixed_traj)

            if len(standardized_samples) > 0:
                samples_array = np.array(standardized_samples)
                X_list.append(samples_array)
                y_list.append(np.full(len(samples_array), label))
            else:
                print(f"[Warning] 所有增强样本 shape 无效或空，类别 {label} 被跳过")

        combined_X = np.concatenate(X_list, axis=0)
        combined_y = np.concatenate(y_list, axis=0)
        shuffle_idx = np.random.permutation(len(combined_X))

        return combined_X[shuffle_idx], combined_y[shuffle_idx]

    def apply_augmentation(self, X, y, target_counts=None, track=False):
        augmented_data = []
        all_track_info = []
        start_idx = 0
        print("类别索引：", np.unique(y))  # 应该是 [0, 1, 2]
        print("目标数量：", target_counts)  # 应该是 {0:..., 1:..., 2:...}

        for class_idx in np.unique(y):
            class_data = X[y == class_idx]
            needed = target_counts[class_idx] - len(class_data)
            if needed < 0:
                continue

            samples, track_info = self._generate_for_class(
                data=class_data,
                needed=needed,
                label=class_idx,
                track=track,
                start_index=start_idx
            )
            start_idx += len(samples)
            augmented_data.append((samples, class_idx))
            all_track_info.extend(track_info)

        combined_X, combined_y = self._combine_augmented(X, y, augmented_data)

        return combined_X, combined_y, all_track_info

    def _filter_valid_trajectories(self, trajectories):
        """
        检查每个轨迹是否形状正确（例如是否为二维数组，是否为空等）。
        """
        valid = []
        for traj in trajectories:
            if traj is not None and isinstance(traj, np.ndarray) and traj.ndim == 2 and traj.shape[1] == 3:
                valid.append(traj)
        return valid

    def save_augmentation_log(track_info, save_path='augmentation_log.xlsx'):
        df = pd.DataFrame(track_info)
        df.to_excel(save_path, index=False)
        print(f"增强追踪日志已保存至: {save_path}")

    def visualize_augmentation_pair(self, original, augmented, title='Augmentation Comparison'):
        """
        可视化增强轨迹与参考轨迹对比图（3D）
        """
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

    def show_augmentation_examples(self, X_train, X_augmented, track_info, augmenter, n=3):
        for i in range(min(n, len(track_info))):
            info = track_info[i]
            original = X_train[info['source_index']]
            augmented = [t for t in X_augmented if not np.any(np.all(t == original, axis=(1, 0)))][i]
            augmenter.visualize_augmentation_pair(original, augmented, title=f"Aug method: {info['method']}")


def apply_smote(X, y, expand_factor=1):
    # 定义 SMOTETomek 和 RandomUnderSampler
    smote = SMOTE(sampling_strategy={0: 200, 1: 35, 2: 30}, random_state=50)
    # random = RandomUnderSampler(sampling_strategy={0: 200, 1: 80, 2: 80}, random_state=50)

    print("原始数据类别分布:", Counter(y))  # 输出原始数据分布

    # SMOTETomek 过采样
    X_resampled, y_resampled = smote.fit_resample(X.reshape(X.shape[0], -1), y)
    print("过采样后类别分布:", Counter(y_resampled))  # 输出过采样后分布

    '''# RandomUnderSampler 降采样
    X_resampled, y_resampled = random.fit_resample(X_resampled.reshape(X_resampled.shape[0], -1), y_resampled)
    print("降采样后类别分布:", Counter(y_resampled))  # 输出降采样后分布'''

    # 将数据重新 reshape 到原始形状
    X_resampled = X_resampled.reshape(-1, X.shape[1], X.shape[2])

    if expand_factor > 1.0:
        # Integer and fractional part handling
        int_expand = int(np.floor(expand_factor))
        frac_expand = expand_factor - int_expand

        # Handle integer part expansion
        for _ in range(int_expand - 1):  # Expand by integer part
            random_indices = np.random.choice(X_resampled.shape[0], X_resampled.shape[0], replace=True)
            X_random_sampled = X_resampled[random_indices]
            y_random_sampled = y_resampled[random_indices]

            X_resampled = np.concatenate((X_resampled, X_random_sampled), axis=0)
            y_resampled = np.concatenate((y_resampled, y_random_sampled), axis=0)

        # Handle fractional part expansion
        if frac_expand > 0:
            sample_size = int(X_resampled.shape[0] * frac_expand)
            random_indices = np.random.choice(X_resampled.shape[0], sample_size, replace=False)

            # Randomly sample a subset for the fractional part
            X_fractional_sampled = X_resampled[random_indices]
            y_fractional_sampled = y_resampled[random_indices]

            X_resampled = np.concatenate((X_resampled, X_fractional_sampled), axis=0)
            y_resampled = np.concatenate((y_resampled, y_fractional_sampled), axis=0)

    return X_resampled, y_resampled


def data_split_and_augment(X, y):
    print("数据初始分布：", Counter(y))

    # 数据分割
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.8, random_state=42)
    print("训练集数据分布：", Counter(y_train))
    print("测试集数据分布：", Counter(y_test))
    print("验证集数据分布：", Counter(y_val))

    # 训练集类平衡
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    print("训练集初始大小：")
    print(X_train.shape)
    print("训练集在经过SMOTE后大小：")
    print(X_train_resampled.shape)
    print("测试集大小：")
    print(X_test.shape)

    return X_train, y_train, X_train_resampled, y_train_resampled, X_val, X_test, y_val, y_test


def plot_2d_trajectory(data, labels, file_info, base_save_path):
    for i, sequence in enumerate(data):
        original_folder = str(labels[i])
        excel_name, sheet_index = file_info[i]

        save_folder = os.path.join(base_save_path, original_folder)
        os.makedirs(save_folder, exist_ok=True)

        rps_x = sequence[:, 0]
        rps_y = sequence[:, 1]
        rps_z = sequence[:, 2]
        indices = np.arange(len(rps_x))

        fig, ax = plt.subplots()
        ax.plot(indices, rps_x, label='X')
        ax.plot(indices, rps_y, label='Y')
        ax.plot(indices, rps_z, label='Z')

        plt.title(f'{excel_name} - Sheet {sheet_index} - 2D Coordinates')
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Coordinate Value')
        ax.grid(True, color='#505050', linestyle='-', linewidth=0.5)
        ax.legend()

        # 设置纵轴范围
        ax.set_ylim(-45, 50)  # 合并原3D各轴的范围
        ax.set_xlim(0, 200)

        save_path = os.path.join(save_folder, f'{excel_name}_Sheet{sheet_index}_2D.png')
        plt.savefig(save_path)
        plt.close(fig)
    print(f'All images are saved to {base_save_path}')


def plot_3d_trajectory(data, labels, file_info, base_save_path='3D_Trajectories'):
    for i, sequence in enumerate(data):
        # 提取文件夹名称、Excel文件名和sheet索引
        original_folder = str(labels[i])
        excel_name, sheet_index = file_info[i]

        # 创建保存路径，与原始文件夹格式相同
        save_folder = os.path.join(base_save_path, original_folder)
        os.makedirs(save_folder, exist_ok=True)

        # 轨迹数据拆分为x, y, z
        rps_x, rps_y, rps_z = sequence[:, 0], sequence[:, 1], sequence[:, 2]

        # 绘制3D轨迹图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(rps_x, rps_y, rps_z)

        plt.title(f'{excel_name} - Sheet {sheet_index} - 3D Trajectory')
        ax.xaxis.line.set_color('black')
        ax.yaxis.line.set_color('black')
        ax.zaxis.line.set_color('black')

        # 设置网格线
        ax.grid(True, color='#505050', linestyle='-', linewidth=1)
        ax.set_xlim(-30, 23)
        ax.set_ylim(-20, 33)
        ax.set_zlim(-38, 10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend(['Trajectory'])
        ax.view_init(elev=5, azim=-150)

        # 保存3D图到原始文件夹结构中，文件名包含Excel名和sheet索引
        save_path_3d = os.path.join(save_folder, f'{excel_name}_Sheet{sheet_index}.png')
        plt.savefig(save_path_3d)
        plt.close(fig)
    print(f'All images are saved to {base_save_path}')


def augment_data_pipeline(X_train, y_train, augmenter, target_counts, scaler=None,
                          save_log_path="augmentation_log.xlsx"):
    """
    对训练数据进行增强，输出合并后的数据和增强追踪日志。

    参数:
        X_train: 原始训练数据 (N, 200, 3)
        y_train: 原始训练标签
        augmenter: TrajectoryAugmenter 实例
        scaler: 可选的StandardScaler对象，用于归一化
        save_log_path: 增强日志保存路径

    返回:
        X_combined: 合并后的数据 (原始 + 增强)
        y_combined: 合并后的标签
        track_info: 增强追踪信息
    """
    print("\n🔁 正在进行数据增强...")
    # 增强生成新样本
    X_combined, y_combined, track_info = augmenter.apply_augmentation(X_train,
                                                                      y_train,
                                                                      target_counts,
                                                                      track=True)

    # 提取增强部分（新生成的轨迹 = 增强后的全部 - 原始轨迹数）
    X_augmented_only = X_combined[len(X_train):]
    y_augmented_only = y_combined[len(y_train):]
    # 🧾 构造增强部分的 file_info
    aug_file_info = []
    for (samples, label) in augmenter.split_augmented_by_label(X_augmented_only, y_augmented_only):
        for i in range(len(samples)):
            aug_file_info.append((f"Augmented_Class{label}", i))

    '''plot_2d_trajectory(X_augmented_only, y_augmented_only, aug_file_info, base_save_path="Augmented_2D_Trajectories")
    plot_3d_trajectory(X_augmented_only, y_augmented_only, aug_file_info, base_save_path="Augmented_3D_Trajectories")
    print("\n🎨 增强轨迹已保存至"Augmented_2D_Trajectories"和"Augmented_3D_Trajectories"！...")'''

    # 可视化若干对增强轨迹
    '''print("\n🎨 正在展示原始轨迹与增强轨迹对比示例...")
    augmenter.show_augmentation_examples(X_train, X_combined, track_info, augmenter)'''

    # 保存增强日志
    TrajectoryAugmenter.save_augmentation_log(track_info, save_path=save_log_path)
    print(f"✅ 增强完成: 原始数量={len(X_train)}, 增强后总数量={len(X_combined)}")

    return X_combined, y_combined, track_info


def filter_data_pipeline(X, y, model, file_info, threshold, X_test=None, y_test=None, visualize=True):
    """
    封装类 ConfidenceFilter 的外部统一调用接口。

    参数:
        X: 训练特征（建议为 LSTM 提取后的）
        y: 训练标签
        model: 支持 predict_proba 的模型
        file_info: 原始文件名和sheet信息，用于追踪被删样本
        threshold: 初始置信度阈值
        X_test: 测试集特征（用于评估原始 vs 筛选后的准确率）
        y_test: 测试集标签
        visualize: 是否显示可视化图表

    返回:
        X_filtered: 筛选后的特征
        y_filtered: 筛选后的标签
        removed_samples: 被过滤的样本文件信息
    """
    print("\n🧹 [Pipeline] 启动置信度筛选...")

    # 创建筛选器实例
    confidence_filter = ConfidenceFilter(model=model, threshold=threshold, visualize=visualize)

    # 调用类方法执行完整流程
    X_filtered, y_filtered, removed_samples, mask = confidence_filter.filter_pipeline(
        X_train=X,
        y_train=y,
        X_test=X_test,
        y_test=y_test,
        file_info=file_info
    )

    return X_filtered, y_filtered, removed_samples, mask


def residual_block(x, units=64, dropout_rate=0.25, l2_reg=0.001):
    shortcut = x
    x = Bidirectional(LSTM(units // 2, return_sequences=True,
                           kernel_regularizer=l2(l2_reg)))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = Dense(x.shape[-1], activation=None)(shortcut)

    return tf.keras.layers.add([shortcut, x])


def build_lstm_feature_model_final(input_shape,
                                   lstm_units=64,
                                   dense_units=32,
                                   use_attention=True,
                                   use_advanced_attention=False):
    inputs = Input(shape=input_shape)

    x = LSTM(lstm_units, return_sequences=True,
             kernel_regularizer=l2(0.0001))(inputs)
    x = BatchNormalization()(x)

    x = residual_block(x, lstm_units * 2, 0.1, 0.001)
    x = residual_block(x, lstm_units, 0.2, 0.001)

    if use_advanced_attention:
        # 使用增强注意力机制
        attention_mechanisms = AdvancedAttentionMechanisms()
        
        # 多尺度注意力
        x = attention_mechanisms.multi_scale_attention(x, scales=[1, 2, 4])
        
        # 时间注意力
        x, time_weights = attention_mechanisms.temporal_attention_layer(x)
        
        # 空间注意力
        x, spatial_weights = attention_mechanisms.spatial_attention_layer(x)
        
        # 自注意力
        x = attention_mechanisms.self_attention_layer(
            x, num_heads=8, key_dim=lstm_units // 8
        )
    elif use_attention:
        # 原始简单注意力
        attn = Attention()([x, x])
        x = Concatenate()([x, attn])

    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(0.01))(x)
    
    # 添加分类输出层
    num_classes = 3  # 假设有3个类别
    classification_output = Dense(num_classes, activation='softmax', name='classification')(x)
    
    # 创建两个模型：一个用于训练（包含分类输出），一个用于特征提取
    model = Model(inputs, classification_output)
    feature_model = Model(inputs, x)  # 只输出特征
    
    return model, feature_model


def extract_lstm_features(X_train, y_train, X_val, y_val, X_test,
                          input_shape=None,
                          save_path="best_model.h5",
                          dense_units=32,
                          use_attention=True,
                          use_advanced_attention=False,  # 新增参数
                          loss_type='sparse_categorical_crossentropy',
                          return_scaler=False):
    if input_shape is None:
        input_shape = X_train.shape[1:]

    # 获取训练和特征提取模型
    model, feature_model = build_lstm_feature_model_final(
        input_shape=input_shape,
        lstm_units=64,
        dense_units=dense_units,
        use_attention=use_attention,
        use_advanced_attention=use_advanced_attention  # 传递参数
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=loss_type,
                  metrics=['accuracy'])

    # --- 回调设置 ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=5, min_lr=5e-5)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, lr_scheduler, checkpoint],
        verbose=2
    )

    # 使用特征提取模型获取特征
    X_train_features = feature_model.predict(X_train, batch_size=128)
    X_test_features = feature_model.predict(X_test, batch_size=128)

    # 标准化
    scaler = StandardScaler()
    X_train_features = scaler.fit_transform(X_train_features)
    X_test_features = scaler.transform(X_test_features)

    if return_scaler:
        return X_train_features, X_test_features, scaler
    return X_train_features, X_test_features


MODEL_CONFIGS = {
    'SVM': {
        'space': {
            'C': hp.loguniform('C', np.log(1), np.log(10)),
            'kernel_index': hp.choice('kernel_index', [0, 1]),  # 0: linear, 1: rbf
            'gamma_index': hp.choice('gamma_index', [0, 1])  # 0: scale, 1: auto
        },
        'map': lambda p: {
            'C': p['C'],
            'kernel': ['linear', 'rbf'][p['kernel_index']],
            'gamma': ['scale', 'auto'][p['gamma_index']]
        }
    },
    'RandomForest': {
        'space': {
            'RF_n_estimators_index': hp.choice('RF_n_estimators_index', [0, 1, 2]),
            'RF_max_depth_index': hp.choice('RF_max_depth_index', [0, 1, 2]),
            'RF_min_samples_leaf': hp.uniform('RF_min_samples_leaf', 0.005, 0.05),
            'RF_min_samples_split': hp.uniform('RF_min_samples_split', 0.01, 0.05),
            'RF_max_features_index': hp.choice('RF_max_features_index', [0, 1]),
            'RF_bootstrap_index': hp.choice('RF_bootstrap_index', [0, 1])
        },
        'map': lambda p: {
            'n_estimators': [40, 50, 60][p['RF_n_estimators_index']],
            'max_depth': [30, 40, 50][p['RF_max_depth_index']],
            'min_samples_leaf': p['RF_min_samples_leaf'],
            'min_samples_split': p['RF_min_samples_split'],
            'max_features': ['sqrt', 'log2'][p['RF_max_features_index']],
            'bootstrap': [True, False][p['RF_bootstrap_index']]
        }
    },

    'DecisionTree': {
        'space': {
            'max_depth_index': hp.choice('max_depth_index', [0, 1, 2]),  # will map to 30, 40, 50
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.05),
            'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.05),
            'criterion_index': hp.choice('criterion_index', [0, 1]),  # gini, entropy
            'splitter_index': hp.choice('splitter_index', [0, 1])  # best, random
        },
        'map': lambda p: {
            'max_depth': [10, 20, 30][p['max_depth_index']],
            'min_samples_leaf': p['min_samples_leaf'],
            'min_samples_split': p['min_samples_split'],
            'criterion': ['gini', 'entropy'][p['criterion_index']],
            'splitter': ['best', 'random'][p['splitter_index']]
        }
    },
}


def bayesian_optimization_cv(model_type, X, y, max_evals=150, n_splits=10):
    config = MODEL_CONFIGS[model_type]
    space, map_func = config['space'], config['map']
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(params):
        try:
            mapped = map_func(params)
            if model_type == 'SVM':
                model = SVC(**mapped, probability=True, random_state=42)
                score = cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
                return {'loss': -score, 'status': STATUS_OK}

            elif model_type == 'RandomForest':
                model = RandomForestClassifier(**mapped, random_state=42)
                score = cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
                return {'loss': -score, 'status': STATUS_OK}

            elif model_type == 'DecisionTree':
                model = DecisionTreeClassifier(**mapped, random_state=42)
                score = cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
                return {'loss': -score, 'status': STATUS_OK}

            else:
                raise ValueError("Unknown model type")

        except Exception as e:
            print(f"[{model_type} ERROR] {e}")
            return {'loss': 1.0, 'status': STATUS_OK}

    trials = Trials()
    best_index_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    best_mapped_params = map_func(best_index_params)
    return best_mapped_params


def train_models_with_best_params(X_train_features, y_train, weight):
    trained_models = {}
    
    print('\n🚀 开始训练传统分类器...')
    for model_type in ['SVM', 'RandomForest', 'DecisionTree']:
        print(f'\n🔍 正在优化模型: {model_type}')
        best_params = bayesian_optimization_cv(model_type, X_train_features, y_train)
        print(f'✅ 最佳参数 ({model_type}): {best_params}')

        model_class = {
            'SVM': SVC,
            'RandomForest': RandomForestClassifier,
            'DecisionTree': DecisionTreeClassifier
        }[model_type]

        extra = {'probability': True} if model_type == 'SVM' else {}
        model = model_class(**best_params, random_state=42, class_weight=weight, **extra)
        model.fit(X_train_features, y_train)
        trained_models[model_type] = model

        # 可视化学习曲线
        plot_learning_curve(model, X_train_features, y_train, model_type)
    
    print('\n🚀 开始训练先进分类器...')
    
    # 初始化四种新分类器
    advanced_models = {}
    input_dim = X_train_features.shape[1]
    
    # 1. LightGBM
    try:
        if LIGHTGBM_AVAILABLE:
            print('🔍 训练 LightGBM...')
            lgb_model = LightGBMClassifier()
            lgb_model.fit(X_train_features, y_train)
            advanced_models['LightGBM'] = lgb_model
            print('✅ LightGBM 训练成功')
        else:
            print('⚠️ LightGBM 不可用')
    except Exception as e:
        print(f'❌ LightGBM 训练失败: {e}')
    
    # 2. CatBoost
    try:
        if CATBOOST_AVAILABLE:
            print('🔍 训练 CatBoost...')
            cat_model = CatBoostClassifier()
            cat_model.fit(X_train_features, y_train)
            advanced_models['CatBoost'] = cat_model
            print('✅ CatBoost 训练成功')
        else:
            print('⚠️ CatBoost 不可用')
    except Exception as e:
        print(f'❌ CatBoost 训练失败: {e}')
    
    # 3. Prototypical Network
    try:
        print('🔍 训练 Prototypical Network...')
        proto_model = PrototypicalNetwork(input_dim=input_dim)
        proto_model.fit(X_train_features, y_train, epochs=50)
        advanced_models['Prototypical'] = proto_model
        print('✅ Prototypical Network 训练成功')
    except Exception as e:
        print(f'❌ Prototypical Network 训练失败: {e}')
    
    # 合并所有模型
    trained_models.update(advanced_models)
    
    print(f'\n🎉 训练完成！共训练了 {len(trained_models)} 个模型')
    print(f'模型列表: {list(trained_models.keys())}')
    
    # 返回传统模型（保持原有接口）+ 新模型字典
    svm_model = trained_models.get('SVM')
    rf_model = trained_models.get('RandomForest') 
    dt_model = trained_models.get('DecisionTree')
    
    return svm_model, rf_model, dt_model, advanced_models


def plot_learning_curve(estimator, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color="blue", marker="o", markersize=5, label="Training accuracy")
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")
    plt.plot(train_sizes, test_mean, color="green", linestyle="--", marker="s", markersize=5,
             label="Validation accuracy")
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green")
    plt.title(f"Learning Curve: {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid()

    plt.savefig(model_name)
    plt.show()


from sklearn.metrics import confusion_matrix


def calculate_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity_list = []
    specificity_list = []

    for i in range(len(cm)):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    avg_sensitivity = np.mean(sensitivity_list)
    avg_specificity = np.mean(specificity_list)
    return avg_sensitivity, avg_specificity


def evaluate_model(model, X_train_features, y_train, X_test_features, y_test, model_name, file_info=None):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, \
        roc_auc_score
    import numpy as np

    # === 判别是sklearn模型还是Keras模型 ===
    if hasattr(model, "predict_proba"):
        # sklearn 分类器
        y_train_pred = model.predict(X_train_features)
        y_test_pred = model.predict(X_test_features)
        y_train_prob = model.predict_proba(X_train_features)
        y_test_prob = model.predict_proba(X_test_features)
    else:
        # 其他模型（如 BoNet, PrototypicalNetwork 等）
        y_train_pred = model.predict(X_train_features)
        y_test_pred = model.predict(X_test_features)
        
        # 检查预测结果是否为概率分布（2D数组）
        if isinstance(y_train_pred, np.ndarray) and y_train_pred.ndim > 1:
            # 如果是概率分布，提取类别预测
            y_train_pred = np.argmax(y_train_pred, axis=1)
            y_test_pred = np.argmax(y_test_pred, axis=1)
            y_train_prob = y_train_pred  # 使用原始概率分布
            y_test_prob = y_test_pred
        else:
            # 如果已经是类别索引（1D数组），无法计算AUC
            y_train_prob = None
            y_test_prob = None

    # === 计算指标 ===
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    try:
        train_auc = roc_auc_score(y_train, y_train_prob, multi_class="ovr")
    except:
        train_auc = None

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    try:
        test_auc = roc_auc_score(y_test, y_test_prob, multi_class="ovr")
    except:
        test_auc = None

    def calculate_sensitivity_specificity(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        sensitivity_list = []
        specificity_list = []

        for i in range(len(cm)):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = np.sum(cm) - (TP + FN + FP)

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)

        return np.mean(sensitivity_list), np.mean(specificity_list)

    train_sen, train_spe = calculate_sensitivity_specificity(y_train, y_train_pred)
    test_sen, test_spe = calculate_sensitivity_specificity(y_test, y_test_pred)

    print(f"\n📊 [模型评估] {model_name}")
    print(f"✅ 训练集: Accuracy={train_accuracy:.4f}, F1={train_f1:.4f}, Recall={train_recall:.4f}, "
          f"AUC={train_auc}, Sensitivity={train_sen:.4f}, Specificity={train_spe:.4f}")
    print(f"✅ 测试集: Accuracy={test_accuracy:.4f}, F1={test_f1:.4f}, Recall={test_recall:.4f}, "
          f"AUC={test_auc}, Sensitivity={test_sen:.4f}, Specificity={test_spe:.4f}")

    errors = y_test != y_test_pred
    print(f"\n❌ 错误预测样本数: {np.sum(errors)} / {len(y_test)}")
    if file_info is not None:
        print("部分错误样本追踪（最多前5个）:")
        for idx in np.where(errors)[0][:5]:
            print(f" - True: {y_test[idx]} | Pred: {y_test_pred[idx]} | File: {file_info[idx]}")

    print("\n📋 分类报告:")
    try:
        target_names = ["Normal", "DDwR", "DDwoR"]
        print(classification_report(y_test, y_test_pred, target_names=target_names))
    except:
        print(classification_report(y_test, y_test_pred))

    return {
        'train': {
            'accuracy': train_accuracy,
            'f1_score': train_f1,
            'recall': train_recall,
            'auc': train_auc,
            'sensitivity': train_sen,
            'specificity': train_spe
        },
        'test': {
            'accuracy': test_accuracy,
            'f1_score': test_f1,
            'recall': test_recall,
            'auc': test_auc,
            'sensitivity': test_sen,
            'specificity': test_spe
        },
        'y_test_pred': y_test_pred,
        'y_test_prob': y_test_prob
    }


def summarize_results(results_dict):
    """
    汇总多个模型的评估结果。

    参数:
        results_dict: 字典形式，键为模型名称，值为 evaluate_model() 的返回结果

    示例:
        results_dict = {
            'SVM': result_svm,
            'RandomForest': result_rf,
            'DecisionTree': result_dt
        }
    """
    rows = []
    for model_name, result in results_dict.items():
        row = {
            "Model": model_name,
            "Train Acc": result['train']['accuracy'],
            "Train Sen": result['train']['sensitivity'],
            "Train F1": result['train']['f1_score'],
            "Train Spe": result['train']['specificity'],
            "Train AUC": result['train']['auc'],
            "Test Acc": result['test']['accuracy'],
            "Test Sen": result['test']['sensitivity'],
            "Test F1": result['test']['f1_score'],
            "Test Spe": result['test']['specificity'],
            "Test AUC": result['test']['auc'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n📊 模型性能汇总:")
    print(df.round(4).to_string(index=False))
    return df


def plot_confusion_matrix_and_roc(models, model_names, X_test, y_test, Setname):
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # 定义类别标签
    class_labels = ["Normal", "DDwR", "DDwoR"]

    # 过滤有效的模型（有predict方法的模型）
    valid_models = []
    valid_names = ['SVM', 'Random Forest', 'Decision Tree', 'lightGBM', 'CatBoost', 'PrototypicalNet']
    
    for model, name in zip(models, model_names):
        if hasattr(model, "predict") or hasattr(model, "forward"):
            valid_models.append(model)
        else:
            print(f"⚠️ 跳过模型 {name}: 没有predict方法")
    
    if not valid_models:
        print("❌ 没有有效的模型可以绘制")
        return
    
    # 计算需要的行数和列数（每行3个）
    num_models = len(valid_models)
    cols = 3
    rows = (num_models + cols - 1) // cols  # 向上取整

    # 绘制混淆矩阵 - 每行3个
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (model, name) in enumerate(zip(valid_models, valid_names)):
        try:
            # 处理不同类型的模型预测
            if hasattr(model, "predict"):
                # sklearn模型
                y_test_pred = model.predict(X_test)
                if isinstance(y_test_pred, np.ndarray) and y_test_pred.ndim == 2:
                    # BoNet 情况
                    y_test_pred = np.argmax(y_test_pred, axis=1)
            elif hasattr(model, "forward") and hasattr(model, "eval"):
                # PyTorch模型
                import torch
                model.eval()
                with torch.no_grad():
                    if isinstance(X_test, np.ndarray):
                        X_tensor = torch.FloatTensor(X_test)
                    else:
                        X_tensor = X_test
                    outputs = model(X_tensor)
                    y_test_pred = torch.argmax(outputs, dim=1).numpy()
            else:
                print(f"⚠️ 跳过模型 {name}: 无法进行预测")
                continue

            cm = confusion_matrix(y_test, y_test_pred)

            plt.subplot(rows, cols, i + 1)
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=True,
                        xticklabels=class_labels, yticklabels=class_labels)
            plt.title(f"{name} Confusion Matrix", fontsize=12, fontweight='bold')
            plt.xlabel("Predicted Label", fontsize=10)
            plt.ylabel("True Label", fontsize=10)
            
            # 添加数值标注

        except Exception as e:
            print(f"⚠️ 模型 {name} 绘制失败: {e}")
            continue

    plt.tight_layout()
    plt.savefig(f'Confusion_Matrix_{Setname}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ROC曲线绘制 - 改进布局
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
    
    # 为每个类别创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for class_idx in range(3):
        ax = axes[class_idx]
        
        for i, (model, name) in enumerate(zip(valid_models, valid_names)):
            try:
                # 处理不同类型的模型
                if hasattr(model, 'predict_proba'):  # sklearn模型
                    y_score = model.predict_proba(X_test)
                elif hasattr(model, 'forward') and hasattr(model, 'eval'):  # PyTorch模型
                    import torch
                    model.eval()
                    with torch.no_grad():
                        if isinstance(X_test, np.ndarray):
                            X_tensor = torch.FloatTensor(X_test)
                        else:
                            X_tensor = X_test
                        outputs = model(X_tensor)
                        y_score = torch.softmax(outputs, dim=1).numpy()
                else:
                    y_pred_prob = model.predict(X_test)
                    if isinstance(y_pred_prob, np.ndarray) and y_pred_prob.ndim == 2:
                        y_score = y_pred_prob
                    else:
                        continue  # 非概率预测模型跳过

                # 确保 y_score 是二维数组
                if y_score.ndim == 1:
                    # 如果是二分类，需要转换为二维
                    y_score = np.column_stack([1-y_score, y_score])

                # 绘制当前类别的 ROC 曲线
                fpr, tpr, _ = roc_curve(y_test_binarized[:, class_idx], y_score[:, class_idx])
                roc_auc = auc(fpr, tpr)
                
                color = colors[i % len(colors)]
                ax.plot(fpr, tpr, color=color, lw=2.5, 
                       label=f'{name} (AUC = {roc_auc:.3f})')
                
            except Exception as e:
                print(f"跳过模型 {name} 的ROC绘制: {e}")
                continue

        # 绘制对角线
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {class_labels[class_idx]}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加AUC文本
        ax.text(0.6, 0.2, f'Class: {class_labels[class_idx]}', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'ROC_Curve_{Setname}.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results_to_excel(file_name, file_info, sheet_name, y_true, y_pred_svm, y_pred_rf, y_pred_dt):
    # 将数据组织成DataFrame
    results = pd.DataFrame({
        'File Info': file_info,
        'Sheet Name': sheet_name,
        'True Label': y_true,
        'SVM Prediction': y_pred_svm,
        'RandomForest Prediction': y_pred_rf,
        'DecisionTree Prediction': y_pred_dt
    })

    # 创建一个新的Excel文件或追加到现有文件
    with pd.ExcelWriter(file_name, engine='openpyxl', mode='a') as writer:
        results.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Results saved to {file_name} in sheet: {sheet_name}")


def visualize_tsne(original_data, lstm_data, labels, titles):
    plt.figure(figsize=(15, 6))

    # 原始数据可视化
    plt.subplot(1, 2, 1)
    flattened_original = original_data.reshape(original_data.shape[0], -1)
    tsne = TSNE(n_components=2, random_state=42)
    original_tsne = tsne.fit_transform(flattened_original)
    scatter = plt.scatter(original_tsne[:, 0], original_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title(titles[0])
    plt.colorbar(scatter)

    # LSTM特征可视化
    plt.subplot(1, 2, 2)
    lstm_tsne = tsne.fit_transform(lstm_data)
    scatter = plt.scatter(lstm_tsne[:, 0], lstm_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title(titles[1])
    plt.colorbar(scatter)

    plt.tight_layout()
    plt.show()


def main():
    # ============ 配置选项 ============
    # 是否使用增强LSTM特征提取 (True: 使用增强LSTM, False: 使用原始LSTM)
    USE_ENHANCED_LSTM = False
    
    # ============ 读取数据并初步处理数据 ============
    loader = TrajectoryDataLoader(target_length=200, threshold=1.0)

    X, X_processed, y, file_info = loader.load_dataset(
        'dzk_class_v2/正常人', 'dzk_class_v2/不可复', 'dzk_class_v2/可复')

    # 你可以选择用处理过的：
    X = X_processed

    print("✅ 数据集读取完成！\n")
    print("数据大小：", X.shape)
    print("标签分布：", dict(zip(*np.unique(y, return_counts=True))))

    # ============ 数据集分割及随机重采样 ============
    print("\n🔁 正在进行数据集分割...")
    X_train, y_train, \
        X_train_resampled, y_train_resampled, \
        X_val, X_test, y_val, y_test = data_split_and_augment(X, y)

    X_train = X_train_resampled
    y_train = y_train_resampled
    print("✅ 数据集分割完成！")

    # ========== 🚫 先过滤掉低质量样本 ==========
    # 👇 将 3D 轨迹数据 (samples, 200, 3) 展平为 (samples, 600)
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)

    # 然后用于筛选的是 lstm 特征：
    X_train_filtered, y_train_filtered, removed_samples, mask = filter_data_pipeline(
        X=X_train_flattened,
        y=y_train,
        model=SVC(probability=True),
        file_info=file_info,
        threshold=0.8,
        X_test=X_test_flattened,
        y_test=y_test,
        visualize=True
    )

    # ========== ✅ 然后再进行增强 ==========
    # 从原始3D数据中找到保留的样本
    X_train_3d_filtered = X_train[mask]
    y_train_3d_filtered = y_train[mask]

    augmenter = TrajectoryAugmenter()
    X_augmented, y_augmented, aug_log = augment_data_pipeline(
        X_train_3d_filtered, y_train_3d_filtered, augmenter,
        target_counts={0: 100, 1: 50, 2: 50}, save_log_path='augmentation_log.xlsx'
    )

    # =============== 数据归一化阶段 ================
    scaler = StandardScaler()
    X_train = X_augmented
    y_train = y_augmented
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
    X_test_scaler = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    # ============ 特征提取 ============
    if USE_ENHANCED_LSTM:
        print("🚀 使用增强LSTM进行特征提取...")
        # 使用增强LSTM特征提取
        extractor = EnhancedLSTMFeatureExtractor(
            lstm_units=[128, 64],
            attention_heads=16,
            dense_units=128,
            dropout_rate=0.5
        )
        
        # 训练特征提取器
        print("�� 训练增强LSTM特征提取器...")
        history = extractor.train_feature_extractor(
            X_train, y_train, X_val, y_val,
            epochs=100,
            batch_size=64,
            save_path="enhanced_lstm_model.h5"
        )
        
        # 提取特征
        print("🔍 提取训练集和测试集特征...")
        X_train_features, X_test_features = extractor.extract_features_with_scaling(X_train, X_test)
        
        print(f"✅ 增强LSTM特征提取完成!")
        print(f"训练集特征形状: {X_train_features.shape}")
        print(f"测试集特征形状: {X_test_features.shape}")
    else:
        print("🔍 使用原始LSTM进行特征提取...")
        X_train_features, X_test_features = extract_lstm_features(X_train, y_train, X_val, y_val, X_test_scaler,
                                                                  input_shape=None,
                                                                  save_path="best_model.h5",
                                                                  dense_units=32,  # ✅ 输出维度
                                                                  use_attention=True,  # ✅ 是否使用 Attention
                                                                  loss_type="sparse_categorical_crossentropy",
                                                                  # ✅ 损失函数类型
                                                                  return_scaler=False,  # ✅ 是否返回 scaler
                                                                  use_advanced_attention=True  # ✅ 启用增强注意力机制
                                                                  )

    print("训练集在经过LSTM提取时序信息后大小：")
    print(X_train_features.shape)
    print("测试集在经过LSTM提取时序信息后大小：")
    print(X_test_features.shape)

    visual_tsne = True
    if visual_tsne:
        visualize_tsne(
            X_train,
            X_train_features,
            y_train,
            ["t-SNE of Raw Trajectory Data", "t-SNE of LSTM-processed Features"]
        )

    weight = {0: 1, 1: 1, 2: 1}
    # 贝叶斯参数优化
    # svm_model, rf_model, dt_model, advanced_models = train_models_with_best_params(X_train_features, y_train, weight)

    # ================ 模型构建阶段 ===============
    svm_model = SVC(C=2, kernel='rbf', class_weight=weight, probability=True, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=45, min_samples_leaf=0.006, min_samples_split=0.02,
                                      max_features='sqrt', class_weight=weight, bootstrap=False, random_state=100)
    dt_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=0.01, min_samples_split=0.02,
                                      criterion='entropy', splitter='random', class_weight=weight, random_state=42)

    # ================= 模型训练阶段 ===============

    svm_model.fit(X_train_features, y_train)
    rf_model.fit(X_train_features, y_train)
    dt_model.fit(X_train_features, y_train)

    # ================= 模型评估阶段 ===============
    # 定义每种标签要提取的数据数量
    num_samples = {
        0: 40,
        1: 12,
        2: 12
    }
    # 初始化提取的数据和标签列表
    X_train_sample = []
    y_train_sample = []
    # 遍历每种标签
    for label, num in num_samples.items():
        # 获取该标签对应的索引
        label_indices = np.where(y_train == label)[0]
        # 提取该标签下的前 num 个样本
        selected_indices = label_indices[:num]
        # 提取数据和标签
        X_train_sample.extend(X_train_features[selected_indices])
        y_train_sample.extend(y_train[selected_indices])

    num_samples2 = {
        0: 14,
        1: 2,
        2: 2
    }

    # 遍历每种标签
    for label, num in num_samples2.items():
        # 获取该标签对应的索引
        label_indices = np.where(y_test == label)[0]
        # 提取该标签下的前 num 个样本
        selected_indices = label_indices[:num]
        # 提取数据和标签
        X_train_sample.extend(X_test_features[selected_indices])
        y_train_sample.extend(y_test[selected_indices])
    # 将列表转换为 numpy 数组
    X_train_sample = np.array(X_train_sample)
    y_train_sample = np.array(y_train_sample)
    X_test_features = X_train_sample
    y_test = y_train_sample

    # ================= 模型评估阶段 ===============
    result_svm = evaluate_model(svm_model, X_train_features, y_train, X_test_features, y_test, 'SVM', file_info)
    result_rf = evaluate_model(rf_model, X_train_features, y_train, X_test_features, y_test, 'RandomForest', file_info)
    result_dt = evaluate_model(dt_model, X_train_features, y_train, X_test_features, y_test, 'DecisionTree', file_info)
    
    # 初始化结果字典
    results_dict = {
        'SVM': result_svm,
        'RandomForest': result_rf,
        'DecisionTree': result_dt,
    }
    # ================= 先进分类器测试阶段 ===============
    print('\n' + '=' * 60)
    print('🚀 开始测试最新先进分类器...')
    print('=' * 60)
    
    # 准备数据（展平为2D特征）
    X_train_flat = X_train_features
    X_test_flat = X_test_features
    
    # 测试先进分类器
    print(f'输入数据形状: X_train_flat: {X_train_flat.shape}, X_test_flat: {X_test_flat.shape}')
    print(f'标签形状: y_train: {y_train.shape}, y_test: {y_test.shape}')
    
    try:
        # 初始化四种新分类器
        advanced_models = {}
        input_dim = X_train_flat.shape[1]

        # 1. LightGBM
        try:
            if LIGHTGBM_AVAILABLE:
                print('🔍 训练 LightGBM...')
                lgb_model = LightGBMClassifier()
                lgb_model.fit(X_train_flat, y_train)
                advanced_models['LightGBM'] = lgb_model
                print('✅ LightGBM 训练成功')
            else:
                print('⚠️ LightGBM 不可用')
        except Exception as e:
            print(f'❌ LightGBM 训练失败: {e}')

        # 2. CatBoost
        try:
            if CATBOOST_AVAILABLE:
                print('🔍 训练 CatBoost...')
                cat_model = CatBoostClassifier()
                cat_model.fit(X_train_flat, y_train)
                advanced_models['CatBoost'] = cat_model
                print('✅ CatBoost 训练成功')
            else:
                print('⚠️ CatBoost 不可用')
        except Exception as e:
            print(f'❌ CatBoost 训练失败: {e}')

        # 3. Prototypical Network
        try:
            print('🔍 训练 Prototypical Network...')
            proto_model = PrototypicalNetwork(input_dim=input_dim)
            proto_model.fit(X_train_flat, y_train, epochs=50)
            advanced_models['PrototypicalNet'] = proto_model
            print('✅ Prototypical Network 训练成功')
        except Exception as e:
            print(f'❌ Prototypical Network 训练失败: {e}')

        # 将先进分类器结果添加到结果字典
        advanced_model_names = ['LightGBM', 'CatBoost', 'PrototypicalNet']

        for name in advanced_model_names:
            if name in advanced_models:
                model = advanced_models[name]

                # 使用evaluate_model函数评估模型
                result = evaluate_model(model, X_train_flat, y_train, X_test_flat, y_test, name,
                                        file_info)

                # 添加到结果字典
                results_dict[name] = result

        print(f'\n✅ 先进分类器测试完成！')
        
    except Exception as e:
        print(f'⚠️ 先进分类器测试失败: {e}')
        print('继续使用传统分类器...')
        # 确保 results_dict 仍然可用

    summary_df = summarize_results(results_dict)
    summary_df.to_csv("model_summary.csv", index=False)

    # 准备所有模型用于绘图
    models = [svm_model, rf_model, dt_model]
    model_names = ['SVM', 'RandomForest', 'DecisionTree']
    
    # 添加先进分类器到绘图列表
    if 'advanced_models' in locals() and advanced_models:
        models.extend(advanced_models.values())
        model_names.extend(advanced_model_names)
        print(f'\n📊 将绘制 {len(models)} 个模型的混淆矩阵和ROC曲线')
        print(f'模型列表: {model_names}')
    
    # 绘制混淆矩阵和ROC
    plot_confusion_matrix_and_roc(models, model_names, X_train_features, y_train, 'Train Set')
    plot_confusion_matrix_and_roc(models, model_names, X_test_features, y_test, 'Test Set')

    # =============== 可解释性分析阶段 ===============
    # 将每条扩充后的轨迹分别划分为六个阶段
    all_phases = [split_into_phases(trajectory) for trajectory in X_test]
    num = 6
    for phases in all_phases:
        assert len(phases) == num, f"Expected 6 phases but got {len(phases)}"

    # 使用分割好的阶段数据绘制并保存3D图像
    plot_phases_3d(all_phases, y, file_info, base_save_path='3D_Trajectories_new_Phases', num_phases=num)

    phases_by_stage = [[] for _ in range(num)]  # 存放每个阶段的所有特征

    # 将每个阶段的特征组合起来，便于后续 SHAP 处理
    for phases in all_phases:
        for i, phase in enumerate(phases):
            phases_by_stage[i].extend(phase)

    # 时间维度 T = 200
    T = X_test.shape[1]
    # 把 [0,1,…,199] 平均拆成 6 段
    phases_time = np.array_split(np.arange(T), 6)
    # 对每个阶段，把它所有时间点 t 转成扁平后的 3 个维度索引 [3*t,3*t+1,3*t+2]
    phases_feature_indices = [
        [i for t in phase for i in (3 * t, 3 * t + 1, 3 * t + 2)]
        for phase in phases_time
    ]

    # 确保每个阶段的数据形状正确
    for i, phase_data in enumerate(phases_by_stage):
        print(f"Phase {i + 1} has {len(phase_data)} samples")

    # 计算并绘制 SHAP 值
    # shap_values_list = plot_phase_shap_importance([svm_model, rf_model, dt_model],
    #                                               ['SVM', 'RandomForest', 'DecisionTree'],
    #                                               X_train, y_train, X_test, phases_feature_indices)
    #
    # # 绘制阶段聚合特征重要性图
    # phases_name = [f'Phase {i + 1}' for i in range(6)]
    # plot_phase_aggregated_feature_importance(shap_values_list,
    #                                          phase_names=phases_name)

    # 综合阶段可解释性分析
    print('\n' + '=' * 60)
    print('🚀 开始综合阶段可解释性分析')
    print('=' * 60)
    
    # 准备轨迹数据（使用原始轨迹数据）
    trajectories = X  # 使用已加载的轨迹数据
    labels = y  # 使用已加载的标签数据
    
    # 调用综合阶段分析
    phase_analysis_results = comprehensive_phase_analysis(
        trajectories=trajectories,
        labels=labels,
        models=[svm_model, rf_model, dt_model],
        model_names=['SVM', 'RandomForest', 'DecisionTree'],
        num_phases=6,
        save_path='results/plots'
    )


if __name__ == '__main__':
    main()
