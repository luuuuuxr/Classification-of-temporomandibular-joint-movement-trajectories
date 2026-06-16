"""
数据处理模块
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.utils.utils import print_data_info, validate_data

class TrajectoryDataLoader:
    """轨迹数据加载器"""
    
    def __init__(self, target_length=200, threshold=1.0):
        self.target_length = target_length
        self.threshold = threshold
    
    def load_dataset(self, normal_path, patient1_path, patient2_path):
        """加载完整数据集"""
        normal_data, normal_labels, normal_info = self._load_folder(normal_path, label=0)
        patient1_data, patient1_labels, patient1_info = self._load_folder(patient1_path, label=1)
        patient2_data, patient2_labels, patient2_info = self._load_folder(patient2_path, label=2)
        
        # 原始轨迹插值填充
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
        """加载单个文件夹的数据"""
        data, labels, file_info = [], [], []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".xlsx"):
                xls = pd.ExcelFile(os.path.join(folder_path, file_name))
                for sheet_index, sheet_name in enumerate(xls.sheet_names):
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=None).iloc[1:]
                    sequence = df.values
                    data.append(sequence)
                    labels.append(label)
                    file_info.append((file_name, sheet_index))
        return data, np.array(labels), file_info
    
    def _truncate_platform(self, trajectory, min_consecutive=2):
        """截断平台期"""
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
        """插值或填充轨迹到目标长度"""
        current_length = len(trajectory)
        if current_length == self.target_length:
            return trajectory
        elif current_length < self.target_length:
            # 插值扩展
            t_old = np.linspace(0, 1, current_length)
            t_new = np.linspace(0, 1, self.target_length)
            interpolated = np.zeros((self.target_length, trajectory.shape[1]))
            for i in range(trajectory.shape[1]):
                f = interp1d(t_old, trajectory[:, i], kind="linear", fill_value="extrapolate")
                interpolated[:, i] = f(t_new)
            return interpolated
        else:
            # 截断
            return trajectory[:self.target_length]
    
    def _process_all_data(self, normal_data, patient1_data, patient2_data):
        """处理所有数据"""
        all_data = normal_data + patient1_data + patient2_data
        processed_data = []
        for traj in all_data:
            truncated = self._truncate_platform(traj)
            interpolated = self._interpolate_or_pad(truncated)
            processed_data.append(interpolated)
        return np.array(processed_data, dtype=np.float32)

def data_split_and_augment(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """数据分割和增强"""
    # 分割数据
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state, stratify=y_temp
    )
    
    # SMOTE过采样
    smote = SMOTE(sampling_strategy={0: 334, 1: 100, 2: 100}, random_state=50)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
    X_train_resampled = X_train_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    
    print_data_info(X_train_resampled, y_train_resampled, "训练集（增强后）")
    print_data_info(X_val, y_val, "验证集")
    print_data_info(X_test, y_test, "测试集")
    
    return X_train, y_train, X_train_resampled, y_train_resampled, X_val, X_test, y_val, y_test
