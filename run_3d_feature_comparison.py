"""
3D轨迹特征工程对比实验（不依赖TensorFlow）
比较传统ML方法在3D特征工程前后的性能
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft, fftfreq
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Trajectory3DFeatureExtractor:
    """3D轨迹特征提取器"""
    
    def __init__(self):
        pass
    
    def extract_3d_spatial_features(self, trajectory):
        """提取3D轨迹的空间几何特征"""
        features = []
        
        # 1. 3D坐标统计特征
        for coord in range(3):  # x, y, z
            coord_data = trajectory[:, coord]
            features.extend([
                np.mean(coord_data),      # 均值
                np.std(coord_data),       # 标准差
                np.ptp(coord_data),       # 极差
                np.median(coord_data),    # 中位数
                np.percentile(coord_data, 25),  # 25分位数
                np.percentile(coord_data, 75),  # 75分位数
            ])
        
        # 2. 运动学特征
        if len(trajectory) > 1:
            velocity = np.diff(trajectory, axis=0)
            speed = np.linalg.norm(velocity, axis=1)
            
            features.extend([
                np.mean(speed),           # 平均速度
                np.std(speed),            # 速度标准差
                np.max(speed),            # 最大速度
                np.min(speed),            # 最小速度
            ])
            
            # 加速度特征
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
        
        # 3. 3D几何特征
        if len(trajectory) > 2:
            # 轨迹总长度
            trajectory_length = np.sum(np.linalg.norm(velocity, axis=1))
            features.append(trajectory_length)
            
            # 直线距离
            straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
            features.append(straight_distance)
            
            # 轨迹复杂度（直线距离/轨迹长度）
            complexity = straight_distance / (trajectory_length + 1e-8)
            features.append(complexity)
            
            # 轨迹弯曲度
            curvature = self._calculate_curvature(trajectory)
            features.append(curvature)
            
            # 3D空间范围
            x_range = np.ptp(trajectory[:, 0])
            y_range = np.ptp(trajectory[:, 1])
            z_range = np.ptp(trajectory[:, 2])
            features.extend([x_range, y_range, z_range])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def _calculate_curvature(self, trajectory):
        """计算轨迹弯曲度"""
        if len(trajectory) < 3:
            return 0.0
        
        # 计算曲率
        first_deriv = np.gradient(trajectory, axis=0)
        second_deriv = np.gradient(first_deriv, axis=0)
        
        # 3D曲率公式
        cross_product = np.cross(first_deriv, second_deriv, axis=1)
        curvature = np.linalg.norm(cross_product, axis=1) / (np.linalg.norm(first_deriv, axis=1) ** 3 + 1e-8)
        
        return np.mean(curvature)
    
    def extract_frequency_features(self, trajectory):
        """提取频域特征"""
        features = []
        
        for coord in range(3):  # x, y, z坐标
            coord_data = trajectory[:, coord]
            
            # FFT变换
            fft_data = fft(coord_data)
            freqs = fftfreq(len(coord_data))
            
            # 只取正频率部分
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft_data[:len(fft_data)//2])
            
            if len(positive_fft) > 0:
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
    
    def extract_all_features(self, X):
        """为所有轨迹提取特征"""
        spatial_features = []
        freq_features = []
        
        for trajectory in X:
            # 提取空间特征
            spatial_feat = self.extract_3d_spatial_features(trajectory)
            spatial_features.append(spatial_feat)
            
            # 提取频域特征
            freq_feat = self.extract_frequency_features(trajectory)
            freq_features.append(freq_feat)
        
        # 合并特征
        spatial_features = np.array(spatial_features)
        freq_features = np.array(freq_features)
        combined_features = np.concatenate([spatial_features, freq_features], axis=1)
        
        return combined_features, spatial_features, freq_features

def load_sample_data():
    """生成模拟3D轨迹数据"""
    print("生成模拟3D轨迹数据...")
    
    np.random.seed(42)
    n_samples = 300
    time_steps = 200
    
    # 正常组：平滑的3D轨迹
    normal_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        x = np.sin(t) + 0.1 * np.random.randn(time_steps)
        y = np.cos(t) + 0.1 * np.random.randn(time_steps)
        z = 0.5 * t + 0.1 * np.random.randn(time_steps)
        trajectory = np.column_stack([x, y, z])
        normal_data.append(trajectory)
    
    # 患者组1：不规则的3D轨迹
    patient1_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        x = np.sin(t) + 0.3 * np.random.randn(time_steps) + 0.2 * np.sin(3*t)
        y = np.cos(t) + 0.3 * np.random.randn(time_steps) + 0.2 * np.cos(3*t)
        z = 0.5 * t + 0.3 * np.random.randn(time_steps) + 0.1 * np.sin(2*t)
        trajectory = np.column_stack([x, y, z])
        patient1_data.append(trajectory)
    
    # 患者组2：更复杂的3D轨迹
    patient2_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        x = np.sin(2*t) + 0.4 * np.random.randn(time_steps) + 0.3 * np.sin(5*t)
        y = np.cos(2*t) + 0.4 * np.random.randn(time_steps) + 0.3 * np.cos(5*t)
        z = 0.3 * t + 0.4 * np.random.randn(time_steps) + 0.2 * np.sin(3*t)
        trajectory = np.column_stack([x, y, z])
        patient2_data.append(trajectory)
    
    # 合并数据
    X = np.array(normal_data + patient1_data + patient2_data)
    y = np.array([0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3))
    
    return X, y

def compare_approaches(X, y, save_path="results"):
    """比较不同方法"""
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 分割数据
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 提取3D特征
    print("\n提取3D特征...")
    extractor = Trajectory3DFeatureExtractor()
    
    # 为训练集提取特征
    spatial_features_train, _, _ = extractor.extract_all_features(X_train)
    spatial_features_val, _, _ = extractor.extract_all_features(X_val)
    spatial_features_test, _, _ = extractor.extract_all_features(X_test)
    
    print(f"3D特征维度: {spatial_features_train.shape[1]}")
    
    # 标准化特征
    scaler = StandardScaler()
    spatial_features_train_scaled = scaler.fit_transform(spatial_features_train)
    spatial_features_val_scaled = scaler.transform(spatial_features_val)
    spatial_features_test_scaled = scaler.transform(spatial_features_test)
    
    # 1. 原始数据（展平后）
    print("\n1. 使用原始数据（展平）...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 标准化原始数据
    scaler_flat = StandardScaler()
    X_train_flat_scaled = scaler_flat.fit_transform(X_train_flat)
    X_val_flat_scaled = scaler_flat.transform(X_val_flat)
    X_test_flat_scaled = scaler_flat.transform(X_test_flat)
    
    # 2. 3D特征工程
    print("2. 使用3D特征工程...")
    
    # 定义模型
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    # 测试原始数据
    print("\n测试原始数据方法...")
    for name, model in models.items():
        # 原始数据
        model.fit(X_train_flat_scaled, y_train)
        val_pred_flat = model.predict(X_val_flat_scaled)
        test_pred_flat = model.predict(X_test_flat_scaled)
        
        val_acc_flat = accuracy_score(y_val, val_pred_flat)
        test_acc_flat = accuracy_score(y_test, test_pred_flat)
        
        results[f"{name} (Raw Data)"] = {
            "val_accuracy": val_acc_flat,
            "test_accuracy": test_acc_flat,
            "predictions": test_pred_flat
        }
        
        print(f"  {name} (Raw): Val={val_acc_flat:.4f}, Test={test_acc_flat:.4f}")
    
    # 测试3D特征
    print("\n测试3D特征工程方法...")
    for name, model in models.items():
        # 3D特征
        model.fit(spatial_features_train_scaled, y_train)
        val_pred_feat = model.predict(spatial_features_val_scaled)
        test_pred_feat = model.predict(spatial_features_test_scaled)
        
        val_acc_feat = accuracy_score(y_val, val_pred_feat)
        test_acc_feat = accuracy_score(y_test, test_pred_feat)
        
        results[f"{name} (3D Features)"] = {
            "val_accuracy": val_acc_feat,
            "test_accuracy": test_acc_feat,
            "predictions": test_pred_feat
        }
        
        print(f"  {name} (3D Features): Val={val_acc_feat:.4f}, Test={test_acc_feat:.4f}")
    
    # 可视化结果
    print("\n生成可视化结果...")
    
    # 1. 性能比较
    plt.figure(figsize=(15, 6))
    
    # 验证集性能
    plt.subplot(1, 2, 1)
    model_names = list(results.keys())
    val_accuracies = [results[name]["val_accuracy"] for name in model_names]
    
    bars = plt.bar(range(len(model_names)), val_accuracies, 
                   color=["skyblue", "lightblue", "lightcyan", "lightgreen", "lightcoral", "lightpink"])
    plt.title("验证集性能比较", fontsize=14, fontweight="bold")
    plt.ylabel("准确率")
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
    plt.ylim(0, 1)
    
    for i, (bar, acc) in enumerate(zip(bars, val_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    
    plt.grid(axis="y", alpha=0.3)
    
    # 测试集性能
    plt.subplot(1, 2, 2)
    test_accuracies = [results[name]["test_accuracy"] for name in model_names]
    
    bars = plt.bar(range(len(model_names)), test_accuracies, 
                   color=["skyblue", "lightblue", "lightcyan", "lightgreen", "lightcoral", "lightpink"])
    plt.title("测试集性能比较", fontsize=14, fontweight="bold")
    plt.ylabel("准确率")
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
    plt.ylim(0, 1)
    
    for i, (bar, acc) in enumerate(zip(bars, test_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 2. 混淆矩阵比较
    plt.figure(figsize=(18, 6))
    
    # 选择最佳模型进行详细比较
    best_raw_model = max([name for name in results.keys() if "Raw Data" in name], 
                        key=lambda x: results[x]["val_accuracy"])
    best_feat_model = max([name for name in results.keys() if "3D Features" in name], 
                         key=lambda x: results[x]["val_accuracy"])
    
    models_to_plot = [best_raw_model, best_feat_model]
    
    for i, model_name in enumerate(models_to_plot):
        plt.subplot(1, 3, i+1)
        cm = confusion_matrix(y_test, results[model_name]["predictions"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Patient1", "Patient2"],
                    yticklabels=["Normal", "Patient1", "Patient2"])
        plt.title(f"{model_name}\nTest Acc: {results[model_name]['test_accuracy']:.3f}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
    
    # 特征重要性（随机森林）
    if "Random Forest" in best_feat_model:
        plt.subplot(1, 3, 3)
        rf_model = models["Random Forest"]
        rf_model.fit(spatial_features_train_scaled, y_train)
        feature_importance = rf_model.feature_importances_
        
        # 选择前20个最重要的特征
        top_indices = np.argsort(feature_importance)[-20:]
        top_importance = feature_importance[top_indices]
        
        plt.barh(range(len(top_importance)), top_importance)
        plt.title("Top 20 Feature Importance\n(Random Forest + 3D Features)")
        plt.xlabel("Importance")
        plt.ylabel("Feature Index")
        plt.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 3. 特征分布可视化
    plt.figure(figsize=(15, 10))
    
    # 选择几个重要的3D特征进行可视化
    important_features = [0, 1, 2, 18, 19, 20]  # 前3个坐标的均值和一些几何特征
    feature_names = ["X_mean", "Y_mean", "Z_mean", "Trajectory_length", "Straight_distance", "Complexity"]
    
    for i, (feat_idx, feat_name) in enumerate(zip(important_features, feature_names)):
        plt.subplot(2, 3, i+1)
        
        for class_label in [0, 1, 2]:
            class_mask = y_train == class_label
            class_data = spatial_features_train_scaled[class_mask, feat_idx]
            plt.hist(class_data, alpha=0.6, label=f"Class {class_label}", bins=20)
        
        plt.title(f"Feature: {feat_name}")
        plt.xlabel("Normalized Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 保存详细结果
    print("\n保存详细结果...")
    
    with open(f"{save_path}/comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("3D轨迹特征工程对比结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 性能排名（按验证集准确率）:\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["val_accuracy"], reverse=True)
        for i, (model, metrics) in enumerate(sorted_results, 1):
            f.write(f"{i}. {model}: Val={metrics['val_accuracy']:.4f}, Test={metrics['test_accuracy']:.4f}\n")
        
        f.write(f"\n2. 最佳模型: {sorted_results[0][0]}\n")
        f.write(f"   验证准确率: {sorted_results[0][1]['val_accuracy']:.4f}\n")
        f.write(f"   测试准确率: {sorted_results[0][1]['test_accuracy']:.4f}\n")
        
        f.write(f"\n3. 3D特征工程效果:\n")
        raw_models = [name for name in results.keys() if "Raw Data" in name]
        feat_models = [name for name in results.keys() if "3D Features" in name]
        
        for raw_model in raw_models:
            base_name = raw_model.replace(" (Raw Data)", "")
            feat_model = f"{base_name} (3D Features)"
            if feat_model in results:
                improvement = results[feat_model]["val_accuracy"] - results[raw_model]["val_accuracy"]
                f.write(f"   {base_name}: {improvement:+.4f} 改善\n")
    
    # 保存分类报告
    with open(f"{save_path}/classification_reports.txt", "w", encoding="utf-8") as f:
        f.write("分类报告\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write(classification_report(y_test, metrics["predictions"], 
                                        target_names=["Normal", "Patient1", "Patient2"]))
            f.write("\n" + "-" * 50 + "\n\n")
    
    print(f"\n所有结果已保存到 {save_path} 文件夹")
    print(f"包含文件:")
    print(f"  - performance_comparison.png: 性能比较图")
    print(f"  - confusion_matrices.png: 混淆矩阵比较图")
    print(f"  - feature_distributions.png: 特征分布图")
    print(f"  - comparison_results.txt: 详细性能结果")
    print(f"  - classification_reports.txt: 分类报告")
    
    return results

def main():
    """主函数"""
    print("=" * 80)
    print("3D轨迹特征工程对比实验")
    print("=" * 80)
    
    # 生成数据
    X, y = load_sample_data()
    print(f"数据形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 运行对比实验
    results = compare_approaches(X, y)
    
    print("\n实验完成！")
    return results

if __name__ == "__main__":
    results = main()
