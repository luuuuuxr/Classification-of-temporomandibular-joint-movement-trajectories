import warnings

import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_phase_shap_importance(models, model_names, X_train, y_train, X_test, phases_by_stage):
    """优化后的跨模型SHAP分析（解决量级差异问题）"""

    # 标准化处理（关键改进点1）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_test_flat = scaler.transform(X_test.reshape(X_test.shape[0], -1))

    # 统一背景样本（关键改进点2）
    background = shap.sample(X_train_flat, min(100, X_train_flat.shape[0]))

    # 多类别处理参数
    class_idx = 1 if len(np.unique(y_train)) > 2 else 'auto'  # 二分类使用auto

    results = []
    for model, name in zip(models, model_names):
        # --- 1) 区分 SKLearn vs Keras ---
        is_keras = False
        # 简单判断：如果 model 有 predict_proba 方法，就认为它是 sklearn；否则当作 Keras
        if name == 'BoNet':
            is_keras = True

        # 先重训练（sklearn 用 fit，Keras BoNet 也可以直接用已经 fit 过的）
        if is_keras:
            # BoNet 通常在外部已经做过 fit，这里不再重复训练
            pass
        else:
            model.fit(X_train_flat, y_train)

        # 统一解释器配置（关键改进点3）
        # 强制统一使用KernelExplainer（技术上可行但存在隐患）
        explainer = shap.KernelExplainer(model.predict_proba, background)
        # 适用于所有模型但效率低下

        # 统一SHAP计算方式
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X_test_flat)

            # 多分类处理（关键改进点4）
            if isinstance(shap_values, list):
                shap_values = np.stack(shap_values).mean(0)  # 多类别取平均
            else:
                shap_values = np.array(shap_values)

        # 1) 按阶段计算平均绝对 SHAP 值 和 平均特征 值
        phase_names = [f"Phase {i + 1}" for i in range(len(phases_by_stage))]

        phase_mean_shap = np.column_stack([
            np.abs(shap_values[:, idx_list]).mean(axis=1)
            for idx_list in phases_by_stage
        ])  # -> (n_test, 6)

        phase_mean_feat = np.column_stack([
            X_test_flat[:, idx_list].mean(axis=1)
            for idx_list in phases_by_stage
        ])  # -> (n_test, 6)

        # 2) 把结果放到 results 里
        results.append((
            name,  # 模型名称
            phase_mean_shap,  # (n_test,6)
            phase_mean_feat,  # (n_test,6)
            phase_names  # ['Phase 1', ..., 'Phase 6']
        ))

        shap.summary_plot(
            phase_mean_shap,
            phase_mean_feat,
            feature_names=phase_names,
            color='#9FD7EF',
            plot_type='bar',
            sort=False,  # ← 禁用自动排序
            show=False
        )

        plt.title(f"{name} SHAP Distribution\n(Mean Absolute Value)")
        plt.gca().set_facecolor('#F5F5F5')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{name}_phase_shap.png")
        plt.show()

    return results


def plot_phase_aggregated_feature_importance(shap_values_list, phase_names):
    num_models = len(shap_values_list)
    num_phases = len(phase_names)
    combined_importance = np.zeros((num_models, num_phases))
    model_names = []

    # 先把每个模型的平均 SHAP 按阶段汇总到 combined_importance
    for i, entry in enumerate(shap_values_list):
        name, phase_shap_values = entry[0], entry[1]
        model_names.append(name)
        # phase_shap_values 可能是 (n_test, n_phases)，我们取它的列平均
        combined_importance[i, :] = phase_shap_values.mean(axis=0)

    # 开始画堆叠柱状图
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    left = np.zeros(num_phases)

    plt.figure(figsize=(8, 6))
    for i in range(num_models):
        plt.barh(
            np.arange(num_phases),
            combined_importance[i],
            height=0.8,
            left=left,
            color=colors[i % len(colors)],
            label=model_names[i],
            alpha=0.8
        )
        left += combined_importance[i]

    plt.yticks(np.arange(num_phases), phase_names)
    plt.gca().invert_yaxis()
    plt.xlabel('Average SHAP Importance')
    plt.title('Overall Importance of Phases Across Models')
    plt.legend(loc='upper right')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('stacked_shap')
    plt.show()
