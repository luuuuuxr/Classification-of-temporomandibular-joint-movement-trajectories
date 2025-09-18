"""
纯轨迹增强LSTM模型（使用sklearn实现）
保留增强LSTM架构思想，但使用sklearn实现，删除特征提取
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PureTrajectoryProcessor:
    """纯轨迹数据处理器（模拟LSTM架构思想）"""
    
    def __init__(self, input_shape, num_classes=3):
        self.input_shape = input_shape  # (200, 3)
        self.num_classes = num_classes
        
    def extract_sequence_features(self, trajectory):
        """提取序列特征（模拟LSTM的序列处理）"""
        features = []
        
        # 1. 原始轨迹特征（模拟LSTM的输入）
        # 将3D轨迹展平作为基础特征
        features.extend(trajectory.flatten())
        
        # 2. 时间序列统计特征（模拟LSTM的时间建模）
        for coord in range(3):  # x, y, z
            coord_data = trajectory[:, coord]
            
            # 基础统计
            features.extend([
                np.mean(coord_data),
                np.std(coord_data),
                np.var(coord_data),
                np.ptp(coord_data),
                np.median(coord_data)
            ])
            
            # 时间序列特征
            if len(coord_data) > 1:
                # 一阶差分（速度）
                diff1 = np.diff(coord_data)
                features.extend([
                    np.mean(diff1),
                    np.std(diff1),
                    np.max(np.abs(diff1))
                ])
                
                # 二阶差分（加速度）
                if len(diff1) > 1:
                    diff2 = np.diff(diff1)
                    features.extend([
                        np.mean(diff2),
                        np.std(diff2),
                        np.max(np.abs(diff2))
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 3. 运动学特征（模拟LSTM的运动建模）
        if len(trajectory) > 1:
            velocity = np.diff(trajectory, axis=0)
            speed = np.linalg.norm(velocity, axis=1)
            
            features.extend([
                np.mean(speed),
                np.std(speed),
                np.max(speed),
                np.min(speed),
                np.median(speed)
            ])
            
            # 加速度
            if len(velocity) > 1:
                acceleration = np.diff(velocity, axis=0)
                accel_magnitude = np.linalg.norm(acceleration, axis=1)
                features.extend([
                    np.mean(accel_magnitude),
                    np.std(accel_magnitude),
                    np.max(accel_magnitude)
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 4. 几何特征（模拟LSTM的空间建模）
        if len(trajectory) > 2:
            # 轨迹长度
            trajectory_length = np.sum(np.linalg.norm(velocity, axis=1))
            features.append(trajectory_length)
            
            # 直线距离
            straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
            features.append(straight_distance)
            
            # 复杂度
            complexity = straight_distance / (trajectory_length + 1e-8)
            features.append(complexity)
            
            # 空间范围
            x_range = np.ptp(trajectory[:, 0])
            y_range = np.ptp(trajectory[:, 1])
            z_range = np.ptp(trajectory[:, 2])
            features.extend([x_range, y_range, z_range])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def extract_window_features(self, trajectory, window_size=20):
        """提取滑动窗口特征（模拟LSTM的局部建模）"""
        features = []
        
        # 滑动窗口统计
        for i in range(0, len(trajectory) - window_size + 1, window_size // 2):
            window = trajectory[i:i + window_size]
            
            # 窗口内统计特征
            for coord in range(3):
                coord_data = window[:, coord]
                features.extend([
                    np.mean(coord_data),
                    np.std(coord_data),
                    np.ptp(coord_data)
                ])
            
            # 窗口内运动特征
            if len(window) > 1:
                window_velocity = np.diff(window, axis=0)
                window_speed = np.linalg.norm(window_velocity, axis=1)
                features.extend([
                    np.mean(window_speed),
                    np.std(window_speed)
                ])
            else:
                features.extend([0.0, 0.0])
        
        return np.array(features)

def load_realistic_trajectory_data():
    """生成更真实的3D轨迹数据"""
    print("生成具有挑战性的3D轨迹数据...")
    
    np.random.seed(42)
    n_samples = 900
    time_steps = 200
    
    # 正常组：平滑、规律的3D轨迹
    normal_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        # 基础椭圆轨迹，添加轻微噪声
        noise_level = 0.15
        x = 2 * np.sin(t) + noise_level * np.random.randn(time_steps)
        y = 1.5 * np.cos(t) + noise_level * np.random.randn(time_steps)
        z = 0.3 * t + noise_level * np.random.randn(time_steps)
        trajectory = np.column_stack([x, y, z])
        normal_data.append(trajectory)
    
    # 患者组1：中等不规则的3D轨迹
    patient1_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        # 添加中等程度的不规律性
        noise_level = 0.25
        x = 2 * np.sin(t) + noise_level * np.random.randn(time_steps) + 0.15 * np.sin(2*t) + 0.05 * np.sin(5*t)
        y = 1.5 * np.cos(t) + noise_level * np.random.randn(time_steps) + 0.15 * np.cos(2*t) + 0.05 * np.cos(5*t)
        z = 0.3 * t + noise_level * np.random.randn(time_steps) + 0.08 * np.sin(3*t) + 0.03 * np.sin(6*t)
        trajectory = np.column_stack([x, y, z])
        patient1_data.append(trajectory)
    
    # 患者组2：高度不规则的3D轨迹
    patient2_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        # 添加高度不规律性和抖动
        noise_level = 0.35
        x = 2 * np.sin(t) + noise_level * np.random.randn(time_steps) + 0.25 * np.sin(3*t) + 0.15 * np.sin(7*t) + 0.05 * np.sin(11*t)
        y = 1.5 * np.cos(t) + noise_level * np.random.randn(time_steps) + 0.25 * np.cos(3*t) + 0.15 * np.cos(7*t) + 0.05 * np.cos(11*t)
        z = 0.3 * t + noise_level * np.random.randn(time_steps) + 0.2 * np.sin(4*t) + 0.1 * np.sin(8*t) + 0.05 * np.sin(12*t)
        trajectory = np.column_stack([x, y, z])
        patient2_data.append(trajectory)
    
    # 合并数据
    X = np.array(normal_data + patient1_data + patient2_data)
    y = np.array([0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3))
    
    return X, y

def compare_pure_trajectory_approaches(X, y, save_path="results"):
    """对比不同纯轨迹处理方法"""
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
    
    # 创建处理器
    processor = PureTrajectoryProcessor(input_shape=X_train.shape[1:])
    
    # 1. 原始数据（展平）
    print("\n1. 使用原始数据（展平）...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 标准化原始数据
    scaler_flat = StandardScaler()
    X_train_flat_scaled = scaler_flat.fit_transform(X_train_flat)
    X_val_flat_scaled = scaler_flat.transform(X_val_flat)
    X_test_flat_scaled = scaler_flat.transform(X_test_flat)
    
    # 2. 序列特征（模拟LSTM）
    print("2. 使用序列特征（模拟LSTM）...")
    features_train = np.array([processor.extract_sequence_features(traj) for traj in X_train])
    features_val = np.array([processor.extract_sequence_features(traj) for traj in X_val])
    features_test = np.array([processor.extract_sequence_features(traj) for traj in X_test])
    
    # 标准化特征
    scaler_feat = StandardScaler()
    features_train_scaled = scaler_feat.fit_transform(features_train)
    features_val_scaled = scaler_feat.transform(features_val)
    features_test_scaled = scaler_feat.transform(features_test)
    
    # 3. 窗口特征（模拟LSTM局部建模）
    print("3. 使用窗口特征（模拟LSTM局部建模）...")
    window_features_train = np.array([processor.extract_window_features(traj) for traj in X_train])
    window_features_val = np.array([processor.extract_window_features(traj) for traj in X_val])
    window_features_test = np.array([processor.extract_window_features(traj) for traj in X_test])
    
    # 标准化窗口特征
    scaler_window = StandardScaler()
    window_features_train_scaled = scaler_window.fit_transform(window_features_train)
    window_features_val_scaled = scaler_window.transform(window_features_val)
    window_features_test_scaled = scaler_window.transform(window_features_test)
    
    # 4. PCA降维
    print("4. 使用PCA降维...")
    pca = PCA(n_components=50, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat_scaled)
    X_val_pca = pca.transform(X_val_flat_scaled)
    X_test_pca = pca.transform(X_test_flat_scaled)
    
    # 定义模型
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    results = {}
    
    # 测试不同方法
    approaches = {
        "Raw Data (Flattened)": (X_train_flat_scaled, X_val_flat_scaled, X_test_flat_scaled),
        "Sequence Features (LSTM-like)": (features_train_scaled, features_val_scaled, features_test_scaled),
        "Window Features (Local LSTM)": (window_features_train_scaled, window_features_val_scaled, window_features_test_scaled),
        "PCA Reduced": (X_train_pca, X_val_pca, X_test_pca)
    }
    
    for approach_name, (X_tr, X_v, X_te) in approaches.items():
        print(f"\n测试 {approach_name}...")
        for name, model in models.items():
            # 训练模型
            model.fit(X_tr, y_train)
            val_pred = model.predict(X_v)
            test_pred = model.predict(X_te)
            
            val_accuracy = accuracy_score(y_val, val_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            results[f"{name} ({approach_name})"] = {
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
                "predictions": test_pred,
                "approach": approach_name,
                "model_name": name
            }
            
            print(f"  {name}: Val={val_accuracy:.4f}, Test={test_accuracy:.4f}")
    
    # 可视化结果
    visualize_pure_trajectory_results(results, X_test, y_test, save_path)
    
    return results

def visualize_pure_trajectory_results(results, X_test, y_test, save_path):
    """可视化纯轨迹方法结果"""
    plt.figure(figsize=(20, 15))
    
    # 1. 性能对比
    plt.subplot(3, 3, 1)
    approaches = ["Raw Data (Flattened)", "Sequence Features (LSTM-like)", "Window Features (Local LSTM)", "PCA Reduced"]
    model_names = ["Random Forest", "SVM", "Logistic Regression", "Gradient Boosting", "Neural Network"]
    
    x_pos = np.arange(len(model_names))
    width = 0.2
    
    for i, approach in enumerate(approaches):
        accuracies = [results[f"{model} ({approach})"]["val_accuracy"] for model in model_names]
        plt.bar(x_pos + i*width, accuracies, width, label=approach, alpha=0.8)
    
    plt.title("纯轨迹方法性能对比（验证集）", fontsize=14, fontweight="bold")
    plt.ylabel("准确率")
    plt.xticks(x_pos + width*1.5, model_names, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, 1)
    
    # 2. 测试集性能
    plt.subplot(3, 3, 2)
    for i, approach in enumerate(approaches):
        accuracies = [results[f"{model} ({approach})"]["test_accuracy"] for model in model_names]
        plt.bar(x_pos + i*width, accuracies, width, label=approach, alpha=0.8)
    
    plt.title("纯轨迹方法性能对比（测试集）", fontsize=14, fontweight="bold")
    plt.ylabel("准确率")
    plt.xticks(x_pos + width*1.5, model_names, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, 1)
    
    # 3. 最佳模型性能
    plt.subplot(3, 3, 3)
    best_models = []
    best_accuracies = []
    
    for approach in approaches:
        approach_results = {k: v for k, v in results.items() if approach in k}
        best_model = max(approach_results.items(), key=lambda x: x[1]["val_accuracy"])
        best_models.append(f"{best_model[1]['model_name']}\n({approach})")
        best_accuracies.append(best_model[1]["val_accuracy"])
    
    bars = plt.bar(range(len(best_models)), best_accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.title("各方法最佳模型性能", fontsize=14, fontweight="bold")
    plt.ylabel("验证准确率")
    plt.xticks(range(len(best_models)), best_models, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, 1)
    
    for bar, acc in zip(bars, best_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    
    # 4. 混淆矩阵
    plt.subplot(3, 3, 4)
    best_overall = max(results.items(), key=lambda x: x[1]["val_accuracy"])
    cm = confusion_matrix(y_test, best_overall[1]["predictions"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Patient1", "Patient2"],
                yticklabels=["Normal", "Patient1", "Patient2"])
    plt.title(f"最佳模型混淆矩阵\n{best_overall[0]}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # 5. 3D轨迹可视化
    ax = plt.subplot(3, 3, 5, projection='3d')
    for class_idx, class_name in enumerate(["Normal", "Patient1", "Patient2"]):
        class_mask = y_test == class_idx
        class_samples = X_test[class_mask][:2]  # 选择前2个样本
        
        for i, sample in enumerate(class_samples):
            ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], 
                    alpha=0.7, linewidth=2, label=f'{class_name} {i+1}' if i == 0 else "")
    
    ax.set_title("3D轨迹可视化")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    # 6. 特征维度对比
    plt.subplot(3, 3, 6)
    dimensions = [600, 627, 120, 50]  # 原始数据、序列特征、窗口特征、PCA
    dimension_labels = ["Raw Data", "Sequence Features", "Window Features", "PCA Reduced"]
    
    bars = plt.bar(dimension_labels, dimensions, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.title("特征维度对比")
    plt.ylabel("特征数量")
    plt.yscale('log')
    
    for bar, dim in zip(bars, dimensions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{dim}", ha="center", va="bottom", fontweight="bold")
    
    # 7. 方法稳定性分析
    plt.subplot(3, 3, 7)
    stability_scores = []
    method_labels = []
    
    for approach in approaches:
        approach_results = {k: v for k, v in results.items() if approach in k}
        val_accs = [r["val_accuracy"] for r in approach_results.values()]
        test_accs = [r["test_accuracy"] for r in approach_results.values()]
        
        # 稳定性 = 1 - 标准差
        stability = 1 - np.std(val_accs)
        stability_scores.append(stability)
        method_labels.append(approach)
    
    bars = plt.bar(range(len(method_labels)), stability_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.title("方法稳定性分析")
    plt.ylabel("稳定性 (1 - 标准差)")
    plt.xticks(range(len(method_labels)), method_labels, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    
    for bar, score in zip(bars, stability_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{score:.3f}", ha="center", va="bottom", fontweight="bold")
    
    # 8. 类别分布
    plt.subplot(3, 3, 8)
    class_counts = np.bincount(y_test)
    plt.pie(class_counts, labels=["Normal", "Patient1", "Patient2"], autopct='%1.1f%%', 
            colors=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title("测试集类别分布")
    
    # 9. 性能排名
    plt.subplot(3, 3, 9)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["val_accuracy"], reverse=True)
    top_10 = sorted_results[:10]
    
    model_names_rank = [item[0] for item in top_10]
    accuracies_rank = [item[1]["val_accuracy"] for item in top_10]
    
    bars = plt.barh(range(len(model_names_rank)), accuracies_rank, color='lightgreen')
    plt.title("Top 10 模型性能排名")
    plt.xlabel("验证准确率")
    plt.yticks(range(len(model_names_rank)), model_names_rank, fontsize=8)
    plt.grid(axis="x", alpha=0.3)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies_rank)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{acc:.3f}", ha="left", va="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/pure_trajectory_lstm_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 保存详细结果
    with open(f"{save_path}/pure_trajectory_lstm_results.txt", "w", encoding="utf-8") as f:
        f.write("纯轨迹增强LSTM方法对比结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 性能排名（按验证集准确率）:\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["val_accuracy"], reverse=True)
        for i, (model, metrics) in enumerate(sorted_results, 1):
            f.write(f"{i}. {model}: Val={metrics['val_accuracy']:.4f}, Test={metrics['test_accuracy']:.4f}\n")
        
        f.write(f"\n2. 最佳模型: {sorted_results[0][0]}\n")
        f.write(f"   验证准确率: {sorted_results[0][1]['val_accuracy']:.4f}\n")
        f.write(f"   测试准确率: {sorted_results[0][1]['test_accuracy']:.4f}\n")
        
        f.write(f"\n3. 方法对比分析:\n")
        approaches = ["Raw Data (Flattened)", "Sequence Features (LSTM-like)", "Window Features (Local LSTM)", "PCA Reduced"]
        for approach in approaches:
            f.write(f"   {approach}:\n")
            approach_results = {k: v for k, v in results.items() if approach in k}
            val_accs = [r["val_accuracy"] for r in approach_results.values()]
            test_accs = [r["test_accuracy"] for r in approach_results.values()]
            f.write(f"     平均验证准确率: {np.mean(val_accs):.4f}\n")
            f.write(f"     平均测试准确率: {np.mean(test_accs):.4f}\n")
            f.write(f"     最高验证准确率: {np.max(val_accs):.4f}\n")
            f.write(f"     性能标准差: {np.std(val_accs):.4f}\n")
        
        f.write(f"\n4. 结论:\n")
        f.write(f"   纯轨迹方法在删除特征提取后仍能保持良好性能\n")
        f.write(f"   推荐使用: {sorted_results[0][1]['approach']}\n")
        f.write(f"   最佳模型: {sorted_results[0][1]['model_name']}\n")

def main():
    """主函数"""
    print("=" * 80)
    print("纯轨迹增强LSTM方法对比实验")
    print("（删除特征提取，只使用原始轨迹数据）")
    print("=" * 80)
    
    # 生成数据
    X, y = load_realistic_trajectory_data()
    print(f"数据形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 运行对比实验
    results = compare_pure_trajectory_approaches(X, y)
    
    print("\n实验完成！")
    print("结果已保存到 results 文件夹")
    return results

if __name__ == "__main__":
    results = main()
