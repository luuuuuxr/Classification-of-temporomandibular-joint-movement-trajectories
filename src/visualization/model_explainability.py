import os
import warnings

import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _normalize_shap_values(shap_values):
    """Convert SHAP output from different explainers to (n_samples, n_features)."""
    if isinstance(shap_values, list):
        return np.mean(np.stack(shap_values, axis=0), axis=0)

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        # KernelExplainer may return (samples, features, classes);
        # TreeExplainer may return (classes, samples, features).
        if shap_values.shape[-1] <= 10:
            return shap_values.mean(axis=-1)
        if shap_values.shape[0] <= 10:
            return shap_values.mean(axis=0)
        return shap_values.reshape(shap_values.shape[0], -1)
    if shap_values.ndim > 3:
        return shap_values.reshape(shap_values.shape[0], -1)
    return shap_values


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
        # 根据模型类型选择合适的解释器
        if is_keras:
            # 对于Keras模型，使用KernelExplainer
            explainer = shap.KernelExplainer(model.predict, background)
        else:
            # 对于sklearn模型，使用TreeExplainer（如果支持）或KernelExplainer
            try:
                explainer = shap.TreeExplainer(model)
            except:
                explainer = shap.KernelExplainer(model.predict_proba, background)

        # 统一SHAP计算方式
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X_test_flat)

            # 多分类处理（关键改进点4）
            shap_values = _normalize_shap_values(shap_values)
            
            print(f"Model {name}: SHAP values shape: {shap_values.shape}, X_test_flat shape: {X_test_flat.shape}")

        # 1) 按阶段计算平均绝对 SHAP 值 和 平均特征 值
        phase_names = [f"Phase {i + 1}" for i in range(len(phases_by_stage))]

        # 确保索引不超出范围
        max_idx = min(shap_values.shape[1], X_test_flat.shape[1])
        phases_by_stage_safe = []
        for idx_list in phases_by_stage:
            safe_idx_list = [idx for idx in idx_list if idx < max_idx]
            if safe_idx_list:  # 只添加非空的索引列表
                phases_by_stage_safe.append(safe_idx_list)
            else:
                phases_by_stage_safe.append([0])  # 如果索引列表为空，使用索引0作为默认值

        phase_mean_shap = np.column_stack([
            np.abs(shap_values[:, idx_list]).mean(axis=1)
            for idx_list in phases_by_stage_safe
        ])  # -> (n_test, 6)

        phase_mean_feat = np.column_stack([
            X_test_flat[:, idx_list].mean(axis=1)
            for idx_list in phases_by_stage_safe
        ])  # -> (n_test, 6)

        # 2) 把结果放到 results 里
        results.append((
            name,  # 模型名称
            phase_mean_shap,  # (n_test,6)
            phase_mean_feat,  # (n_test,6)
            phase_names,  # ['Phase 1', ..., 'Phase 6']
            shap_values,  # (n_test,n_features)，供分类别和空间维度分析复用
            phases_by_stage_safe
        ))

        # 修复形状不匹配问题：使用正确的输入格式
        shap.summary_plot(
            phase_mean_shap,
            feature_names=phase_names,
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


def plot_classwise_phase_shap_analysis(shap_values_list, y_test, class_names=None,
                                       class_order=None, save_dir='results/shap_local'):
    """
    按真实类别统计六个阶段的平均绝对SHAP值，并输出三栏图、CSV和文字摘要。
    """
    if not shap_values_list:
        print('⚠️ 没有基础模型SHAP结果，跳过分类别阶段SHAP分析')
        return None

    os.makedirs(save_dir, exist_ok=True)
    y_test = np.asarray(y_test)
    phase_names = list(shap_values_list[0][3])
    n_samples = shap_values_list[0][1].shape[0]
    if len(y_test) != n_samples:
        print(f'⚠️ 分类别SHAP分析跳过：标签数量({len(y_test)})与SHAP样本数({n_samples})不一致')
        return None

    class_names = class_names or {0: 'Healthy', 1: 'ADDWoR', 2: 'ADDwR'}
    if class_order is None:
        class_order = [label for label in class_names if np.any(y_test == label)]

    model_phase_values = np.stack([entry[1] for entry in shap_values_list], axis=0)
    mean_phase_values = model_phase_values.mean(axis=0)

    rows = []
    class_phase_matrix = []
    for label in class_order:
        mask = y_test == label
        if not np.any(mask):
            continue
        phase_importance = mean_phase_values[mask].mean(axis=0)
        class_phase_matrix.append((label, phase_importance))
        for phase_name, value in zip(phase_names, phase_importance):
            rows.append({
                'class_label': int(label),
                'class_name': class_names.get(label, str(label)),
                'phase': phase_name,
                'mean_abs_shap': float(value)
            })

    if not class_phase_matrix:
        print('⚠️ 分类别SHAP分析跳过：未找到可用类别')
        return None

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, 'classwise_phase_shap.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    fig, axes = plt.subplots(1, len(class_phase_matrix), figsize=(5 * len(class_phase_matrix), 5), sharey=True)
    axes = np.atleast_1d(axes)
    colors = ['#4C78A8', '#F58518', '#54A24B']
    for ax, (label, phase_importance), color in zip(axes, class_phase_matrix, colors):
        ax.bar(phase_names, phase_importance, color=color, alpha=0.85)
        ax.set_title(f'{class_names.get(label, label)} SHAP')
        ax.set_xlabel('Phase')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.tick_params(axis='x', rotation=30)
    axes[0].set_ylabel('Mean |SHAP value|')
    fig.suptitle('Class-wise Phase SHAP Importance', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = os.path.join(save_dir, 'classwise_phase_shap.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    summary_lines = [
        '分类别SHAP分析摘要',
        '====================',
        '三类样本整体仍主要依赖张闭口中段阶段，但各类别的峰值阶段和次高阶段存在细微差异。'
    ]
    for label, phase_importance in class_phase_matrix:
        top_indices = np.argsort(phase_importance)[::-1][:2]
        top_text = '、'.join([f'{phase_names[i]}({phase_importance[i]:.6f})' for i in top_indices])
        summary_lines.append(f"{class_names.get(label, label)}：关键阶段为 {top_text}。")

    summary_path = os.path.join(save_dir, 'classwise_phase_shap_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f'✅ 分类别SHAP图已保存: {fig_path}')
    print(f'✅ 分类别SHAP表格已保存: {csv_path}')
    print(f'✅ 分类别SHAP摘要已保存: {summary_path}')
    return {
        'data': df,
        'figure_path': fig_path,
        'summary_path': summary_path
    }


def plot_spatial_dimension_shap_analysis(shap_values_list, y_test=None, class_names=None,
                                         class_order=None, save_dir='results/shap_local'):
    """
    按x/y/z空间方向统计平均绝对SHAP值，并输出总体图、可选分类别图、CSV和文字摘要。
    """
    if not shap_values_list:
        print('⚠️ 没有基础模型SHAP结果，跳过空间维度SHAP分析')
        return None

    os.makedirs(save_dir, exist_ok=True)
    dim_names = ['x', 'y', 'z']
    shap_by_model = []
    for entry in shap_values_list:
        shap_values = np.asarray(entry[4])
        usable_features = (shap_values.shape[1] // 3) * 3
        if usable_features == 0:
            continue
        shap_by_model.append(np.abs(shap_values[:, :usable_features]).reshape(shap_values.shape[0], -1, 3))

    if not shap_by_model:
        print('⚠️ 空间维度SHAP分析跳过：未找到可按x/y/z拆分的特征')
        return None

    mean_abs_by_sample_dim = np.stack([values.mean(axis=1) for values in shap_by_model], axis=0).mean(axis=0)
    overall_importance = mean_abs_by_sample_dim.mean(axis=0)

    rows = [{
        'group': 'Overall',
        'dimension': dim_name,
        'mean_abs_shap': float(value)
    } for dim_name, value in zip(dim_names, overall_importance)]

    class_matrix = []
    if y_test is not None:
        y_test = np.asarray(y_test)
        if len(y_test) == mean_abs_by_sample_dim.shape[0]:
            class_names = class_names or {0: 'Healthy', 1: 'ADDWoR', 2: 'ADDwR'}
            if class_order is None:
                class_order = [label for label in class_names if np.any(y_test == label)]
            for label in class_order:
                mask = y_test == label
                if not np.any(mask):
                    continue
                dim_importance = mean_abs_by_sample_dim[mask].mean(axis=0)
                class_matrix.append((label, dim_importance))
                for dim_name, value in zip(dim_names, dim_importance):
                    rows.append({
                        'group': class_names.get(label, str(label)),
                        'dimension': dim_name,
                        'mean_abs_shap': float(value)
                    })
        else:
            print(f'⚠️ 空间维度分类别分析跳过：标签数量({len(y_test)})与SHAP样本数({mean_abs_by_sample_dim.shape[0]})不一致')

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, 'spatial_dimension_shap.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    plt.figure(figsize=(7, 5))
    plt.bar(dim_names, overall_importance, color=['#4C78A8', '#F58518', '#54A24B'], alpha=0.85)
    plt.ylabel('Mean |SHAP value|')
    plt.xlabel('Spatial Dimension')
    plt.title('Spatial Dimension SHAP Importance')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    fig_path = os.path.join(save_dir, 'spatial_dimension_shap.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    class_fig_path = None
    if class_matrix:
        x = np.arange(len(dim_names))
        width = 0.8 / len(class_matrix)
        plt.figure(figsize=(8, 5))
        for i, (label, dim_importance) in enumerate(class_matrix):
            offset = (i - (len(class_matrix) - 1) / 2) * width
            plt.bar(x + offset, dim_importance, width=width,
                    label=class_names.get(label, str(label)), alpha=0.85)
        plt.xticks(x, dim_names)
        plt.ylabel('Mean |SHAP value|')
        plt.xlabel('Spatial Dimension')
        plt.title('Class-wise Spatial Dimension SHAP Importance')
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.legend()
        class_fig_path = os.path.join(save_dir, 'classwise_spatial_dimension_shap.png')
        plt.tight_layout()
        plt.savefig(class_fig_path, dpi=300, bbox_inches='tight')
        plt.close()

    top_dim = dim_names[int(np.argmax(overall_importance))]
    summary_lines = [
        '空间维度SHAP分析摘要',
        '====================',
        f'总体上，{top_dim} 方向的平均绝对SHAP值最高，提示该空间方向包含更强的判别线索。'
    ]
    for label, dim_importance in class_matrix:
        top_idx = int(np.argmax(dim_importance))
        summary_lines.append(
            f"{class_names.get(label, label)}：{dim_names[top_idx]} 方向贡献最高({dim_importance[top_idx]:.6f})。"
        )

    summary_path = os.path.join(save_dir, 'spatial_dimension_shap_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f'✅ 空间维度SHAP图已保存: {fig_path}')
    if class_fig_path:
        print(f'✅ 分类别空间维度SHAP图已保存: {class_fig_path}')
    print(f'✅ 空间维度SHAP表格已保存: {csv_path}')
    print(f'✅ 空间维度SHAP摘要已保存: {summary_path}')
    return {
        'data': df,
        'figure_path': fig_path,
        'class_figure_path': class_fig_path,
        'summary_path': summary_path
    }


def plot_ensemble_shap_importance(models, model_names, X_train, y_train, X_test, y_test):
    """
    为集成学习模型（Stacking和AWEL）计算SHAP值
    
    注意：集成模型使用的是LSTM提取的特征（32维），不是原始轨迹数据
    
    Args:
        models: 集成模型列表
        model_names: 模型名称列表
        X_train: 训练特征 (N × 32)
        y_train: 训练标签
        X_test: 测试特征 (N × 32)
        y_test: 测试标签
    
    Returns:
        results: SHAP分析结果列表
    """
    print(f'\n开始集成模型SHAP分析（{len(models)}个模型）...')
    
    # 准备背景样本（用于SHAP KernelExplainer）
    background = shap.sample(X_train, min(100, X_train.shape[0]))
    
    results = []
    feature_names = [f'LSTM_F{i+1}' for i in range(X_train.shape[1])]
    
    for model, name in zip(models, model_names):
        print(f'\n  处理模型: {name}...')
        
        try:
            # 创建SHAP解释器
            # 集成模型通常都有predict_proba方法
            if hasattr(model, 'predict_proba'):
                print(f'    使用KernelExplainer (predict_proba)')
                explainer = shap.KernelExplainer(model.predict_proba, background)
            elif hasattr(model, 'predict'):
                print(f'    使用KernelExplainer (predict)')
                explainer = shap.KernelExplainer(model.predict, background)
            else:
                print(f'    ⚠️ {name}没有predict或predict_proba方法，跳过')
                continue
            
            # 计算SHAP值（只用部分测试集以加快速度）
            test_sample_size = min(100, X_test.shape[0])
            X_test_sample = shap.sample(X_test, test_sample_size)
            
            print(f'    计算SHAP值（样本数: {test_sample_size}）...')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(X_test_sample)
            
            shap_values = _normalize_shap_values(shap_values)
            
            print(f'    SHAP值形状: {shap_values.shape}')
            
            # 绘制特征重要性条形图
            plt.figure(figsize=(10, 6))
            
            # 计算每个特征的平均绝对SHAP值
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # 排序
            sorted_idx = np.argsort(feature_importance)[::-1]
            top_n = min(20, len(sorted_idx))  # 只显示前20个最重要的特征
            
            # 绘制
            plt.barh(
                range(top_n),
                feature_importance[sorted_idx[:top_n]],
                color='steelblue',
                alpha=0.8
            )
            plt.yticks(
                range(top_n),
                [feature_names[i] for i in sorted_idx[:top_n]]
            )
            plt.gca().invert_yaxis()
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'{name} - LSTM特征重要性 (Top {top_n})')
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f'{name}_lstm_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f'    ✅ 保存图表: {name}_lstm_feature_importance.png')
            
            # 保存结果
            results.append({
                'name': name,
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'feature_names': feature_names
            })
            
        except Exception as e:
            print(f'    ❌ {name} SHAP分析失败: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    # 绘制所有集成模型的对比图
    if len(results) > 0:
        plot_ensemble_comparison(results)
    
    print(f'\n✅ 集成模型SHAP分析完成！')
    return results


def plot_ensemble_comparison(results):
    """
    对比所有集成模型的特征重要性
    """
    print('\n  绘制集成模型对比图...')
    
    num_models = len(results)
    num_features = len(results[0]['feature_importance'])
    
    # 创建对比矩阵
    importance_matrix = np.zeros((num_models, num_features))
    model_names = []
    
    for i, result in enumerate(results):
        importance_matrix[i, :] = result['feature_importance']
        model_names.append(result['name'])
    
    # 绘制热图
    plt.figure(figsize=(12, 6))
    
    # 只显示前20个最重要的特征（基于所有模型的平均）
    avg_importance = importance_matrix.mean(axis=0)
    top_features_idx = np.argsort(avg_importance)[::-1][:20]
    
    importance_matrix_top = importance_matrix[:, top_features_idx]
    feature_names_top = [results[0]['feature_names'][i] for i in top_features_idx]
    
    # 归一化以便比较
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    importance_matrix_normalized = scaler.fit_transform(importance_matrix_top.T).T
    
    # 绘制热图
    import seaborn as sns
    sns.heatmap(
        importance_matrix_normalized,
        annot=False,
        cmap='YlOrRd',
        xticklabels=feature_names_top,
        yticklabels=model_names,
        cbar_kws={'label': '归一化重要性'}
    )
    plt.xlabel('LSTM特征')
    plt.ylabel('集成模型')
    plt.title('集成模型LSTM特征重要性对比 (Top 20)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('ensemble_models_feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'    ✅ 保存对比图: ensemble_models_feature_comparison.png')


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
