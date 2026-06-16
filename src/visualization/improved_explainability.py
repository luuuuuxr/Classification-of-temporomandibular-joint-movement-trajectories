import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler


def comprehensive_phase_analysis(trajectories, labels, models, model_names,
                                 num_phases=6, save_path="results/plots"):
    """
    综合阶段可解释性分析
    
    Args:
        trajectories: 轨迹数据
        labels: 标签
        models: 训练好的模型列表
        model_names: 模型名称列表
        num_phases: 阶段数量
        save_path: 保存路径
    """
    print("🚀 开始综合阶段可解释性分析...")

    n_samples = len(trajectories)
    
    print(f"轨迹数据形状: {trajectories.shape}")
    print(f"标签数据形状: {labels.shape}")

    # 分析每个模型
    results = {}

    for model, name in zip(models, model_names):
        print(f"🤖 分析模型: {name}")

        if not hasattr(model, "predict"):
            print(f"模型 {name} 没有 predict 方法，跳过")
            continue

        # 处理轨迹数据：将3D轨迹数据(样本数, 时间步, 3)展平为2D特征(样本数, 时间步*3)
        X_phase = trajectories.reshape(n_samples, -1)
        print(f"展平后特征形状: {X_phase.shape}")
        n_features = X_phase.shape[1]

        # 标准化特征
        scaler = StandardScaler()
        X_phase_scaled = scaler.fit_transform(X_phase)

        try:
            y_pred = model.predict(X_phase_scaled)
            y_pred_class = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
            accuracy = np.mean(y_pred_class == labels)
            print(f"模拟特征准确率: {accuracy:.4f}")

            # 分析每个阶段的重要性
            phase_importance = {}
            
            # 假设轨迹数据时间步为200，将时间维度分成6个阶段
            time_steps = trajectories.shape[1]
            phases_time = np.array_split(np.arange(time_steps), num_phases)
            
            # 计算每个阶段对应的特征索引（每个时间步有3个特征维度）
            phases_feature_indices = [
                [i for t in phase for i in (3 * t, 3 * t + 1, 3 * t + 2)]
                for phase in phases_time
            ]
            
            for phase_idx in range(num_phases):
                phase_features = X_phase_scaled[:, phases_feature_indices[phase_idx]]
                feature_variance = np.var(phase_features, axis=0)
                phase_importance[f"Phase_{phase_idx + 1}"] = {
                    "mean_variance": np.mean(feature_variance),
                    "max_variance": np.max(feature_variance),
                    "feature_count": len(phases_feature_indices[phase_idx])
                }

            results[name] = {
                "accuracy": accuracy,
                "phase_importance": phase_importance,
                "predictions": y_pred_class
            }
            
        except Exception as e:
            print(f"模型 {name} 预测失败: {e}")
            continue

    # 可视化结果
    if results:
        plot_phase_importance_comparison(results, num_phases, save_path)

        # 生成临床洞察
        first_model = list(results.keys())[0]
        phase_importance = results[first_model]["phase_importance"]
        generate_clinical_insights(phase_importance)

    return results


def plot_phase_importance_comparison(results, num_phases, save_path):
    """绘制阶段重要性比较图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 模型准确率比较
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]

    axes[0, 0].bar(model_names, accuracies, color=["blue", "green", "red"])
    axes[0, 0].set_title("Model Accuracy on Phase Features")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0, 1)

    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 0.01, f"{acc:.3f}", ha="center")

    # 2. 阶段重要性热图
    phase_names = [f"Phase_{i + 1}" for i in range(num_phases)]
    importance_matrix = np.zeros((len(model_names), len(phase_names)))

    for i, name in enumerate(model_names):
        for j, phase in enumerate(phase_names):
            importance_matrix[i, j] = results[name]["phase_importance"][phase]["mean_variance"]

    im = axes[0, 1].imshow(importance_matrix, cmap="Blues", aspect="auto")
    axes[0, 1].set_title("Phase Importance Heatmap")
    axes[0, 1].set_xlabel("Phase")
    axes[0, 1].set_ylabel("Model")
    axes[0, 1].set_xticks(range(len(phase_names)))
    axes[0, 1].set_xticklabels(phase_names)
    axes[0, 1].set_yticks(range(len(model_names)))
    axes[0, 1].set_yticklabels(model_names)
    plt.colorbar(im, ax=axes[0, 1])

    # 3. 平均阶段重要性
    mean_importance = np.mean(importance_matrix, axis=0)
    axes[1, 0].bar(phase_names, mean_importance, color="skyblue")
    axes[1, 0].set_title("Average Phase Importance Across Models")
    axes[1, 0].set_ylabel("Mean Variance")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 4. 阶段重要性分布
    phase_importance_data = []
    phase_labels = []

    for phase in phase_names:
        for name in model_names:
            phase_importance_data.append(
                results[name]["phase_importance"][phase]["mean_variance"]
            )
            phase_labels.append(phase)

    phase_df = pd.DataFrame({
        "Phase": phase_labels,
        "Importance": phase_importance_data
    })

    sns.boxplot(data=phase_df, x="Phase", y="Importance", ax=axes[1, 1])
    axes[1, 1].set_title("Phase Importance Distribution")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/phase_importance_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_clinical_insights(phase_importance):
    """生成临床洞察分析报告"""
    print("\n🏥 临床洞察分析报告")
    print("=" * 50)

    # 阶段重要性排序
    phase_names = list(phase_importance.keys())
    importance_values = [phase_importance[phase]["mean_variance"] for phase in phase_names]

    sorted_phases = sorted(zip(phase_names, importance_values),
                           key=lambda x: x[1], reverse=True)

    print("\n📊 阶段重要性排序（基于特征方差）:")
    for i, (phase, importance) in enumerate(sorted_phases, 1):
        print(f"{i}. {phase}: {importance:.4f}")

    print("\n🔬 临床解释:")
    print("• 张口初始阶段 (Phase 1): 反映下颌开始运动的协调性")
    print("• 张口过程阶段 (Phase 2): 反映张口过程中的运动稳定性")
    print("• 张口结束阶段 (Phase 3): 反映最大张口时的关节位置")
    print("• 闭口初始阶段 (Phase 4): 反映闭口开始时的运动控制")
    print("• 闭口过程阶段 (Phase 5): 反映闭口过程中的运动协调")
    print("• 闭口结束阶段 (Phase 6): 反映回到初始位置的运动精度")

    print("\n�� 诊断建议:")
    if sorted_phases[0][0] in ["Phase_1", "Phase_2"]:
        print("• 重点关注张口初期运动，可能存在关节盘移位")
    elif sorted_phases[0][0] in ["Phase_4", "Phase_5"]:
        print("• 重点关注闭口期运动，可能存在肌肉功能异常")
    else:
        print("• 运动异常分布较为均匀，建议综合评估")
