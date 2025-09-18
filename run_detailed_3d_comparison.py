"""
更真实的3D轨迹特征工程对比实验
使用更复杂的模拟数据来展示3D特征工程的优势
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

class AdvancedTrajectory3DFeatureExtractor:
    """高级3D轨迹特征提取器"""
    
    def __init__(self):
        pass
    
    def extract_comprehensive_3d_features(self, trajectory):
        """提取全面的3D轨迹特征"""
        features = []
        
        # 1. 基础统计特征
        for coord in range(3):  # x, y, z
            coord_data = trajectory[:, coord]
            features.extend([
                np.mean(coord_data),      # 均值
                np.std(coord_data),       # 标准差
                np.var(coord_data),       # 方差
                np.ptp(coord_data),       # 极差
                np.median(coord_data),    # 中位数
                np.percentile(coord_data, 25),  # 25分位数
                np.percentile(coord_data, 75),  # 75分位数
                np.skew(coord_data),      # 偏度
                np.kurtosis(coord_data),  # 峰度
            ])
        
        # 2. 运动学特征
        if len(trajectory) > 1:
            velocity = np.diff(trajectory, axis=0)
            speed = np.linalg.norm(velocity, axis=1)
            
            features.extend([
                np.mean(speed),           # 平均速度
                np.std(speed),            # 速度标准差
                np.var(speed),            # 速度方差
                np.max(speed),            # 最大速度
                np.min(speed),            # 最小速度
                np.median(speed),         # 速度中位数
                np.percentile(speed, 25), # 速度25分位数
                np.percentile(speed, 75), # 速度75分位数
            ])
            
            # 加速度特征
            if len(velocity) > 1:
                acceleration = np.diff(velocity, axis=0)
                accel_magnitude = np.linalg.norm(acceleration, axis=1)
                features.extend([
                    np.mean(accel_magnitude),  # 平均加速度
                    np.std(accel_magnitude),   # 加速度标准差
                    np.max(accel_magnitude),   # 最大加速度
                    np.min(accel_magnitude),   # 最小加速度
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * 12)
        
        # 3. 几何特征
        if len(trajectory) > 2:
            # 轨迹长度
            trajectory_length = np.sum(np.linalg.norm(velocity, axis=1))
            features.append(trajectory_length)
            
            # 直线距离
            straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
            features.append(straight_distance)
            
            # 轨迹复杂度
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
            
            # 3D空间体积（近似）
            volume_approx = x_range * y_range * z_range
            features.append(volume_approx)
            
            # 轨迹中心
            center = np.mean(trajectory, axis=0)
            features.extend(center.tolist())
            
            # 轨迹重心到原点的距离
            center_distance = np.linalg.norm(center)
            features.append(center_distance)
            
        else:
            features.extend([0.0] * 12)
        
        # 4. 频域特征
        freq_features = self._extract_frequency_features(trajectory)
        features.extend(freq_features)
        
        # 5. 时域特征
        time_features = self._extract_time_domain_features(trajectory)
        features.extend(time_features)
        
        return np.array(features)
    
    def _calculate_curvature(self, trajectory):
        """计算轨迹弯曲度"""
        if len(trajectory) < 3:
            return 0.0
        
        try:
            first_deriv = np.gradient(trajectory, axis=0)
            second_deriv = np.gradient(first_deriv, axis=0)
            
            cross_product = np.cross(first_deriv, second_deriv, axis=1)
            curvature = np.linalg.norm(cross_product, axis=1) / (np.linalg.norm(first_deriv, axis=1) ** 3 + 1e-8)
            
            return np.mean(curvature)
        except:
            return 0.0
    
    def _extract_frequency_features(self, trajectory):
        """提取频域特征"""
        features = []
        
        for coord in range(3):  # x, y, z坐标
            coord_data = trajectory[:, coord]
            
            try:
                fft_data = fft(coord_data)
                freqs = fftfreq(len(coord_data))
                
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = np.abs(fft_data[:len(fft_data)//2])
                
                if len(positive_fft) > 0:
                    features.extend([
                        np.max(positive_fft),                    # 最大幅值
                        np.mean(positive_fft),                   # 平均幅值
                        np.std(positive_fft),                    # 幅值标准差
                        positive_freqs[np.argmax(positive_fft)], # 主频率
                        np.sum(positive_fft),                    # 总能量
                        np.sum(positive_fft**2),                 # 总功率
                    ])
                else:
                    features.extend([0.0] * 6)
            except:
                features.extend([0.0] * 6)
        
        return features
    
    def _extract_time_domain_features(self, trajectory):
        """提取时域特征"""
        features = []
        
        for coord in range(3):  # x, y, z坐标
            coord_data = trajectory[:, coord]
            
            # 零交叉率
            zero_crossings = np.sum(np.diff(np.sign(coord_data - np.mean(coord_data))) != 0)
            zero_crossing_rate = zero_crossings / len(coord_data)
            features.append(zero_crossing_rate)
            
            # 自相关特征
            if len(coord_data) > 1:
                autocorr = np.correlate(coord_data, coord_data, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
                features.extend([
                    np.max(autocorr[1:]),  # 最大自相关（排除lag=0）
                    np.mean(autocorr[1:]), # 平均自相关
                ])
            else:
                features.extend([0.0, 0.0])
        
        return features
    
    def extract_all_features(self, X):
        """为所有轨迹提取特征"""
        all_features = []
        
        for trajectory in X:
            features = self.extract_comprehensive_3d_features(trajectory)
            all_features.append(features)
        
        return np.array(all_features)

def load_realistic_data():
    """生成更真实的3D轨迹数据"""
    print("生成更真实的3D轨迹数据...")
    
    np.random.seed(42)
    n_samples = 600  # 增加样本数量
    time_steps = 200
    
    # 正常组：平滑、规律的3D轨迹
    normal_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        # 基础椭圆轨迹
        x = 2 * np.sin(t) + 0.1 * np.random.randn(time_steps)
        y = 1.5 * np.cos(t) + 0.1 * np.random.randn(time_steps)
        z = 0.3 * t + 0.1 * np.random.randn(time_steps)
        trajectory = np.column_stack([x, y, z])
        normal_data.append(trajectory)
    
    # 患者组1：轻微不规则的3D轨迹
    patient1_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        # 添加轻微的不规律性
        x = 2 * np.sin(t) + 0.2 * np.random.randn(time_steps) + 0.1 * np.sin(2*t)
        y = 1.5 * np.cos(t) + 0.2 * np.random.randn(time_steps) + 0.1 * np.cos(2*t)
        z = 0.3 * t + 0.2 * np.random.randn(time_steps) + 0.05 * np.sin(3*t)
        trajectory = np.column_stack([x, y, z])
        patient1_data.append(trajectory)
    
    # 患者组2：明显不规则的3D轨迹
    patient2_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        # 添加明显的不规律性和抖动
        x = 2 * np.sin(t) + 0.4 * np.random.randn(time_steps) + 0.3 * np.sin(3*t) + 0.1 * np.sin(7*t)
        y = 1.5 * np.cos(t) + 0.4 * np.random.randn(time_steps) + 0.3 * np.cos(3*t) + 0.1 * np.cos(7*t)
        z = 0.3 * t + 0.4 * np.random.randn(time_steps) + 0.2 * np.sin(4*t) + 0.1 * np.sin(8*t)
        trajectory = np.column_stack([x, y, z])
        patient2_data.append(trajectory)
    
    # 合并数据
    X = np.array(normal_data + patient1_data + patient2_data)
    y = np.array([0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3))
    
    return X, y

def compare_approaches_detailed(X, y, save_path="results"):
    """详细的对比实验"""
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
    print("\n提取高级3D特征...")
    extractor = AdvancedTrajectory3DFeatureExtractor()
    
    # 为训练集提取特征
    features_train = extractor.extract_all_features(X_train)
    features_val = extractor.extract_all_features(X_val)
    features_test = extractor.extract_all_features(X_test)
    
    print(f"3D特征维度: {features_train.shape[1]}")
    
    # 标准化特征
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_val_scaled = scaler.transform(features_val)
    features_test_scaled = scaler.transform(features_test)
    
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
        model.fit(features_train_scaled, y_train)
        val_pred_feat = model.predict(features_val_scaled)
        test_pred_feat = model.predict(features_test_scaled)
        
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
    
    colors = ["skyblue", "lightblue", "lightcyan", "lightgreen", "lightcoral", "lightpink"]
    bars = plt.bar(range(len(model_names)), val_accuracies, color=colors)
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
    
    bars = plt.bar(range(len(model_names)), test_accuracies, color=colors)
    plt.title("测试集性能比较", fontsize=14, fontweight="bold")
    plt.ylabel("准确率")
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
    plt.ylim(0, 1)
    
    for i, (bar, acc) in enumerate(zip(bars, test_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/detailed_performance_comparison.png", dpi=300, bbox_inches="tight")
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
        rf_model.fit(features_train_scaled, y_train)
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
    plt.savefig(f"{save_path}/detailed_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 3. 特征分布可视化
    plt.figure(figsize=(20, 15))
    
    # 选择几个重要的3D特征进行可视化
    important_features = [0, 1, 2, 27, 28, 29, 30, 31, 32]  # 前3个坐标的均值和一些几何特征
    feature_names = ["X_mean", "Y_mean", "Z_mean", "Trajectory_length", "Straight_distance", 
                    "Complexity", "Curvature", "X_range", "Y_range", "Z_range"]
    
    for i, (feat_idx, feat_name) in enumerate(zip(important_features, feature_names)):
        plt.subplot(3, 3, i+1)
        
        for class_label in [0, 1, 2]:
            class_mask = y_train == class_label
            class_data = features_train_scaled[class_mask, feat_idx]
            plt.hist(class_data, alpha=0.6, label=f"Class {class_label}", bins=20)
        
        plt.title(f"Feature: {feat_name}")
        plt.xlabel("Normalized Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/detailed_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 4. 3D轨迹可视化
    fig = plt.figure(figsize=(15, 5))
    
    for class_idx, class_name in enumerate(["Normal", "Patient1", "Patient2"]):
        ax = fig.add_subplot(1, 3, class_idx + 1, projection='3d')
        
        # 选择该类别的几个样本进行可视化
        class_mask = y_train == class_idx
        class_samples = X_train[class_mask][:5]  # 选择前5个样本
        
        for i, sample in enumerate(class_samples):
            ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], 
                   alpha=0.7, linewidth=2, label=f'Sample {i+1}' if i < 3 else "")
        
        ax.set_title(f'{class_name} Trajectories')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if class_idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/3d_trajectory_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 保存详细结果
    print("\n保存详细结果...")
    
    with open(f"{save_path}/detailed_comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("详细3D轨迹特征工程对比结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 性能排名（按验证集准确率）:\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["val_accuracy"], reverse=True)
        for i, (model, metrics) in enumerate(sorted_results, 1):
            f.write(f"{i}. {model}: Val={metrics['val_accuracy']:.4f}, Test={metrics['test_accuracy']:.4f}\n")
        
        f.write(f"\n2. 最佳模型: {sorted_results[0][0]}\n")
        f.write(f"   验证准确率: {sorted_results[0][1]['val_accuracy']:.4f}\n")
        f.write(f"   测试准确率: {sorted_results[0][1]['test_accuracy']:.4f}\n")
        
        f.write(f"\n3. 3D特征工程效果分析:\n")
        raw_models = [name for name in results.keys() if "Raw Data" in name]
        feat_models = [name for name in results.keys() if "3D Features" in name]
        
        f.write("   原始数据 vs 3D特征工程:\n")
        for raw_model in raw_models:
            base_name = raw_model.replace(" (Raw Data)", "")
            feat_model = f"{base_name} (3D Features)"
            if feat_model in results:
                val_improvement = results[feat_model]["val_accuracy"] - results[raw_model]["val_accuracy"]
                test_improvement = results[feat_model]["test_accuracy"] - results[raw_model]["test_accuracy"]
                f.write(f"   {base_name}:\n")
                f.write(f"     验证集改善: {val_improvement:+.4f}\n")
                f.write(f"     测试集改善: {test_improvement:+.4f}\n")
        
        f.write(f"\n4. 特征维度分析:\n")
        f.write(f"   原始数据维度: {X_train_flat.shape[1]} (展平后的3D坐标)\n")
        f.write(f"   3D特征维度: {features_train.shape[1]} (提取的几何和统计特征)\n")
        f.write(f"   维度压缩比: {X_train_flat.shape[1] / features_train.shape[1]:.2f}\n")
    
    # 保存分类报告
    with open(f"{save_path}/detailed_classification_reports.txt", "w", encoding="utf-8") as f:
        f.write("详细分类报告\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write(classification_report(y_test, metrics["predictions"], 
                                        target_names=["Normal", "Patient1", "Patient2"]))
            f.write("\n" + "-" * 60 + "\n\n")
    
    print(f"\n所有结果已保存到 {save_path} 文件夹")
    print(f"包含文件:")
    print(f"  - detailed_performance_comparison.png: 详细性能比较图")
    print(f"  - detailed_confusion_matrices.png: 详细混淆矩阵比较图")
    print(f"  - detailed_feature_distributions.png: 详细特征分布图")
    print(f"  - 3d_trajectory_visualization.png: 3D轨迹可视化图")
    print(f"  - detailed_comparison_results.txt: 详细性能结果")
    print(f"  - detailed_classification_reports.txt: 详细分类报告")
    
    return results

def main():
    """主函数"""
    print("=" * 80)
    print("详细3D轨迹特征工程对比实验")
    print("=" * 80)
    
    # 生成数据
    X, y = load_realistic_data()
    print(f"数据形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 运行对比实验
    results = compare_approaches_detailed(X, y)
    
    print("\n实验完成！")
    return results

if __name__ == "__main__":
    results = main()
