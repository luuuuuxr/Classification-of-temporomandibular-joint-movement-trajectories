"""
纯轨迹数据架构对比实验（使用sklearn）
不使用3D特征工程，只使用原始轨迹数据对比不同架构
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

def extract_basic_trajectory_features(trajectory):
    """提取基础轨迹特征（不使用复杂的3D特征工程）"""
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
    else:
        features.extend([0.0] * 4)
    
    return np.array(features)

def compare_trajectory_approaches(X, y, save_path="results"):
    """对比不同轨迹处理方法"""
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
    
    # 2. 基础特征工程
    print("2. 使用基础特征工程...")
    features_train = np.array([extract_basic_trajectory_features(traj) for traj in X_train])
    features_val = np.array([extract_basic_trajectory_features(traj) for traj in X_val])
    features_test = np.array([extract_basic_trajectory_features(traj) for traj in X_test])
    
    # 标准化特征
    scaler_feat = StandardScaler()
    features_train_scaled = scaler_feat.fit_transform(features_train)
    features_val_scaled = scaler_feat.transform(features_val)
    features_test_scaled = scaler_feat.transform(features_test)
    
    # 3. PCA降维
    print("3. 使用PCA降维...")
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
        "Basic Features": (features_train_scaled, features_val_scaled, features_test_scaled),
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
    
    # 1. 性能比较可视化
    print("\n生成性能比较图...")
    plt.figure(figsize=(20, 12))
    
    # 按方法分组
    approaches = ["Raw Data (Flattened)", "Basic Features", "PCA Reduced"]
    model_names = list(models.keys())
    
    # 验证集性能
    plt.subplot(2, 3, 1)
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    for i, approach in enumerate(approaches):
        accuracies = [results[f"{model} ({approach})"]["val_accuracy"] for model in model_names]
        plt.bar(x_pos + i*width, accuracies, width, label=approach, alpha=0.8)
    
    plt.title("验证集性能比较", fontsize=14, fontweight="bold")
    plt.ylabel("准确率")
    plt.xticks(x_pos + width, model_names, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, 1)
    
    # 测试集性能
    plt.subplot(2, 3, 2)
    for i, approach in enumerate(approaches):
        accuracies = [results[f"{model} ({approach})"]["test_accuracy"] for model in model_names]
        plt.bar(x_pos + i*width, accuracies, width, label=approach, alpha=0.8)
    
    plt.title("测试集性能比较", fontsize=14, fontweight="bold")
    plt.ylabel("准确率")
    plt.xticks(x_pos + width, model_names, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, 1)
    
    # 最佳模型性能对比
    plt.subplot(2, 3, 3)
    best_models = []
    best_accuracies = []
    
    for approach in approaches:
        approach_results = {k: v for k, v in results.items() if approach in k}
        best_model = max(approach_results.items(), key=lambda x: x[1]["val_accuracy"])
        best_models.append(f"{best_model[1]['model_name']}\n({approach})")
        best_accuracies.append(best_model[1]["val_accuracy"])
    
    bars = plt.bar(range(len(best_models)), best_accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title("各方法最佳模型性能", fontsize=14, fontweight="bold")
    plt.ylabel("验证准确率")
    plt.xticks(range(len(best_models)), best_models, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, 1)
    
    for bar, acc in zip(bars, best_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    
    # 混淆矩阵比较
    plt.subplot(2, 3, 4)
    best_overall = max(results.items(), key=lambda x: x[1]["val_accuracy"])
    cm = confusion_matrix(y_test, best_overall[1]["predictions"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Patient1", "Patient2"],
                yticklabels=["Normal", "Patient1", "Patient2"])
    plt.title(f"最佳模型混淆矩阵\n{best_overall[0]}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # 特征重要性（随机森林）
    plt.subplot(2, 3, 5)
    rf_models = {k: v for k, v in results.items() if "Random Forest" in k}
    if rf_models:
        best_rf = max(rf_models.items(), key=lambda x: x[1]["val_accuracy"])
        # 重新训练以获取特征重要性
        approach_name = best_rf[1]["approach"]
        if approach_name == "Raw Data (Flattened)":
            X_data = X_train_flat_scaled
        elif approach_name == "Basic Features":
            X_data = features_train_scaled
        else:  # PCA Reduced
            X_data = X_train_pca
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_data, y_train)
        feature_importance = rf_model.feature_importances_
        
        # 选择前20个最重要的特征
        top_indices = np.argsort(feature_importance)[-20:]
        top_importance = feature_importance[top_indices]
        
        plt.barh(range(len(top_importance)), top_importance)
        plt.title(f"Top 20 Feature Importance\n{best_rf[0]}")
        plt.xlabel("Importance")
        plt.ylabel("Feature Index")
        plt.grid(axis="x", alpha=0.3)
    
    # 3D轨迹可视化
    ax = plt.subplot(2, 3, 6, projection='3d')
    for class_idx, class_name in enumerate(["Normal", "Patient1", "Patient2"]):
        class_mask = y_train == class_idx
        class_samples = X_train[class_mask][:3]  # 选择前3个样本
        
        for i, sample in enumerate(class_samples):
            ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], 
                    alpha=0.7, linewidth=2, label=f'{class_name} {i+1}' if i == 0 else "")
    
    ax.set_title("3D轨迹可视化")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/trajectory_approaches_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 2. 详细性能分析
    print("生成详细性能分析图...")
    plt.figure(figsize=(20, 15))
    
    # 性能热力图
    plt.subplot(3, 3, 1)
    performance_matrix = np.zeros((len(approaches), len(model_names)))
    for i, approach in enumerate(approaches):
        for j, model in enumerate(model_names):
            key = f"{model} ({approach})"
            performance_matrix[i, j] = results[key]["val_accuracy"]
    
    sns.heatmap(performance_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=model_names, yticklabels=approaches)
    plt.title("性能热力图（验证集准确率）")
    plt.xlabel("模型")
    plt.ylabel("方法")
    
    # 泛化能力分析
    plt.subplot(3, 3, 2)
    generalization_gaps = []
    model_labels = []
    for approach in approaches:
        for model in model_names:
            key = f"{model} ({approach})"
            val_acc = results[key]["val_accuracy"]
            test_acc = results[key]["test_accuracy"]
            gap = abs(val_acc - test_acc)
            generalization_gaps.append(gap)
            model_labels.append(f"{model}\n({approach})")
    
    plt.bar(range(len(generalization_gaps)), generalization_gaps, color='lightcoral')
    plt.title("泛化能力分析（|验证准确率 - 测试准确率|）")
    plt.ylabel("准确率差异")
    plt.xticks(range(len(model_labels)), model_labels, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    
    # 方法对比雷达图
    plt.subplot(3, 3, 3)
    metrics = ['平均准确率', '最高准确率', '稳定性', '一致性']
    
    approach_scores = {}
    for approach in approaches:
        approach_results = {k: v for k, v in results.items() if approach in k}
        val_accs = [r["val_accuracy"] for r in approach_results.values()]
        test_accs = [r["test_accuracy"] for r in approach_results.values()]
        
        avg_acc = np.mean(val_accs)
        max_acc = np.max(val_accs)
        stability = 1 - np.std(val_accs)  # 稳定性 = 1 - 标准差
        consistency = 1 - np.mean([abs(v-t) for v, t in zip(val_accs, test_accs)])  # 一致性
        
        approach_scores[approach] = [avg_acc, max_acc, stability, consistency]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for approach, scores in approach_scores.items():
        scores += scores[:1]  # 闭合
        plt.plot(angles, scores, 'o-', linewidth=2, label=approach)
        plt.fill(angles, scores, alpha=0.25)
    
    plt.xticks(angles[:-1], metrics)
    plt.ylim(0, 1)
    plt.title("方法综合对比雷达图")
    plt.legend()
    plt.grid(True)
    
    # 特征维度分析
    plt.subplot(3, 3, 4)
    dimensions = [X_train_flat.shape[1], features_train.shape[1], X_train_pca.shape[1]]
    dimension_labels = ["Raw Data", "Basic Features", "PCA Reduced"]
    
    bars = plt.bar(dimension_labels, dimensions, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title("特征维度对比")
    plt.ylabel("特征数量")
    plt.yscale('log')
    
    for bar, dim in zip(bars, dimensions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{dim}", ha="center", va="bottom", fontweight="bold")
    
    # 训练时间估算（基于模型复杂度）
    plt.subplot(3, 3, 5)
    model_complexity = {
        "Random Forest": 100,
        "SVM": 50,
        "Logistic Regression": 10,
        "Gradient Boosting": 80,
        "Neural Network": 60
    }
    
    complexity_scores = []
    model_names_list = []
    for model in model_names:
        complexity_scores.append(model_complexity[model])
        model_names_list.append(model)
    
    plt.bar(model_names_list, complexity_scores, color='lightblue')
    plt.title("模型复杂度估算")
    plt.ylabel("相对复杂度")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    
    # 性能vs复杂度散点图
    plt.subplot(3, 3, 6)
    for approach in approaches:
        val_accs = []
        complexities = []
        for model in model_names:
            key = f"{model} ({approach})"
            val_accs.append(results[key]["val_accuracy"])
            complexities.append(model_complexity[model])
        
        plt.scatter(complexities, val_accs, s=100, alpha=0.7, label=approach)
    
    plt.xlabel("模型复杂度")
    plt.ylabel("验证准确率")
    plt.title("性能 vs 复杂度")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 类别分布
    plt.subplot(3, 3, 7)
    class_counts = np.bincount(y_train)
    plt.pie(class_counts, labels=["Normal", "Patient1", "Patient2"], autopct='%1.1f%%', 
            colors=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title("训练集类别分布")
    
    # 特征分布示例
    plt.subplot(3, 3, 8)
    # 选择几个重要特征进行可视化
    important_features = [0, 1, 2, 15, 16, 17]  # 前3个坐标的均值和一些运动特征
    feature_names = ["X_mean", "Y_mean", "Z_mean", "Speed_mean", "Speed_std", "Speed_max"]
    
    for i, (feat_idx, feat_name) in enumerate(zip(important_features, feature_names)):
        if feat_idx < features_train.shape[1]:
            for class_label in [0, 1, 2]:
                class_mask = y_train == class_label
                class_data = features_train_scaled[class_mask, feat_idx]
                plt.hist(class_data, alpha=0.6, label=f"Class {class_label}", bins=20)
            break  # 只显示第一个特征
    
    plt.title(f"特征分布示例: {feature_names[0]}")
    plt.xlabel("Normalized Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 模型性能排名
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
    plt.savefig(f"{save_path}/trajectory_detailed_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 保存详细结果
    print("\n保存详细结果...")
    
    with open(f"{save_path}/trajectory_comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("纯轨迹数据架构对比结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 性能排名（按验证集准确率）:\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["val_accuracy"], reverse=True)
        for i, (model, metrics) in enumerate(sorted_results, 1):
            f.write(f"{i}. {model}: Val={metrics['val_accuracy']:.4f}, Test={metrics['test_accuracy']:.4f}\n")
        
        f.write(f"\n2. 最佳模型: {sorted_results[0][0]}\n")
        f.write(f"   验证准确率: {sorted_results[0][1]['val_accuracy']:.4f}\n")
        f.write(f"   测试准确率: {sorted_results[0][1]['test_accuracy']:.4f}\n")
        
        f.write(f"\n3. 方法对比分析:\n")
        for approach in approaches:
            f.write(f"   {approach}:\n")
            approach_results = {k: v for k, v in results.items() if approach in k}
            val_accs = [r["val_accuracy"] for r in approach_results.values()]
            test_accs = [r["test_accuracy"] for r in approach_results.values()]
            f.write(f"     平均验证准确率: {np.mean(val_accs):.4f}\n")
            f.write(f"     平均测试准确率: {np.mean(test_accs):.4f}\n")
            f.write(f"     最高验证准确率: {np.max(val_accs):.4f}\n")
            f.write(f"     性能标准差: {np.std(val_accs):.4f}\n")
        
        f.write(f"\n4. 特征维度分析:\n")
        f.write(f"   原始数据维度: {X_train_flat.shape[1]} (展平后的3D坐标)\n")
        f.write(f"   基础特征维度: {features_train.shape[1]} (提取的统计特征)\n")
        f.write(f"   PCA降维维度: {X_train_pca.shape[1]} (主成分分析)\n")
        
        f.write(f"\n5. 结论:\n")
        best_overall = max(results.items(), key=lambda x: x[1]["val_accuracy"])
        f.write(f"   最佳方法: {best_overall[1]['approach']}\n")
        f.write(f"   最佳模型: {best_overall[1]['model_name']}\n")
        f.write(f"   最佳性能: {best_overall[1]['val_accuracy']:.4f}\n")
    
    # 保存分类报告
    with open(f"{save_path}/trajectory_classification_reports.txt", "w", encoding="utf-8") as f:
        f.write("纯轨迹数据分类报告\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name}:\n")
            f.write(classification_report(y_test, result["predictions"], 
                                        target_names=["Normal", "Patient1", "Patient2"]))
            f.write("\n" + "-" * 60 + "\n\n")
    
    print(f"\n所有结果已保存到 {save_path} 文件夹")
    print(f"包含文件:")
    print(f"  - trajectory_approaches_comparison.png: 方法对比图")
    print(f"  - trajectory_detailed_analysis.png: 详细分析图")
    print(f"  - trajectory_comparison_results.txt: 性能结果")
    print(f"  - trajectory_classification_reports.txt: 分类报告")
    
    return results

def main():
    """主函数"""
    print("=" * 80)
    print("纯轨迹数据架构对比实验")
    print("=" * 80)
    
    # 生成数据
    X, y = load_realistic_trajectory_data()
    print(f"数据形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 运行对比实验
    results = compare_trajectory_approaches(X, y)
    
    print("\n实验完成！")
    return results

if __name__ == "__main__":
    results = main()
