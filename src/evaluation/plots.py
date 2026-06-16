# -*- coding: utf-8 -*-
"""
评估可视化模块：
  plot_param_search_diagnostics / plot_learning_curve /
  plot_confusion_matrix_and_roc / visualize_tsne
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize


# ──────────────────────────────────────────────────────────────
#  调参诊断图
# ──────────────────────────────────────────────────────────────

def plot_param_search_diagnostics(model_type, trial_history,
                                  save_dir="results/param_tuning"):
    if not trial_history:
        return
    os.makedirs(save_dir, exist_ok=True)
    valid_trials = [t for t in trial_history if t.get('score') is not None]
    if not valid_trials:
        return

    scores = [t['score'] for t in valid_trials]
    iterations = np.arange(1, len(scores) + 1)
    best_idx = int(np.argmax(scores))
    best_iter = iterations[best_idx]
    best_score = scores[best_idx]

    if model_type in ['LightGBM', 'CatBoost', 'PrototypicalNet']:
        fig = plt.figure(figsize=(14, 10))
        bottom_margin = 0.08
    else:
        fig = plt.figure(figsize=(13, 9.5))
        bottom_margin = 0.06

    fig.suptitle(
        f'{model_type} Hyperparameter Tuning  |  Best Acc: {best_score:.3f} (Iter {best_iter})',
        fontsize=16, y=0.98)

    param_df = pd.DataFrame([t['params'] for t in valid_trials])
    numeric_df = param_df.select_dtypes(include=[np.number])
    numeric_cols = numeric_df.columns.tolist()
    num_params = len(numeric_cols)

    if not numeric_df.empty:
        if num_params == 1:
            single_ax = fig.add_subplot(111)
            param_name = numeric_cols[0]
            sns.scatterplot(x=numeric_df[param_name], y=scores, hue=scores,
                            palette='viridis', ax=single_ax, s=60,
                            edgecolor='white', linewidth=0.5)
            x_vals = numeric_df[param_name].values
            if len(x_vals) >= 3:
                sorted_idx = np.argsort(x_vals)
                x_sorted = x_vals[sorted_idx]
                y_sorted = np.array(scores)[sorted_idx]
                try:
                    coeffs = np.polyfit(x_sorted, y_sorted,
                                        deg=min(3, len(x_sorted) - 1))
                    poly = np.poly1d(coeffs)
                    x_line = np.linspace(x_sorted.min(), x_sorted.max(), 200)
                    single_ax.plot(x_line, poly(x_line), color='#F25F5C', linewidth=2)
                except Exception:
                    pass
            single_ax.set_xlabel(param_name)
            single_ax.set_ylabel('Accuracy')
            single_ax.set_title('Parameter-response relationship')
            single_ax.legend([], [], frameon=False)

        elif num_params <= 4:
            pair_spec = fig.add_gridspec(
                num_params, num_params, wspace=0.15, hspace=0.2,
                left=0.08, right=0.92, top=0.92, bottom=bottom_margin + 0.05)
            cmap = plt.cm.viridis
            acc_norm = ((np.array(scores) - np.min(scores))
                        / (np.max(scores) - np.min(scores) + 1e-12))
            for i, row_name in enumerate(numeric_cols):
                for j, col_name in enumerate(numeric_cols):
                    ax = fig.add_subplot(pair_spec[i, j])
                    if i == j:
                        sns.kdeplot(numeric_df[col_name], ax=ax,
                                    fill=True, color='#0B4F6C', alpha=0.6)
                        ax.set_ylabel('')
                    elif i > j:
                        ax.scatter(numeric_df[col_name], numeric_df[row_name],
                                   c=acc_norm, cmap=cmap, s=25, alpha=0.75,
                                   edgecolor='none')
                    else:
                        ax.axis('off')
                    if i < num_params - 1:
                        ax.set_xticklabels([])
                    else:
                        ax.set_xlabel(col_name, rotation=30, ha='right', fontsize=9)
                        ax.tick_params(axis='x', labelsize=8)
                    if j > 0:
                        ax.set_yticklabels([])
                    else:
                        ax.set_ylabel(row_name, fontsize=9)
                        ax.tick_params(axis='y', labelsize=8)
            cbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.25])
            plt.colorbar(plt.cm.ScalarMappable(
                norm=plt.Normalize(0, 1), cmap=cmap), cax=cbar_ax)
            cbar_ax.set_ylabel('Accuracy (normalized)', fontsize=10)

        else:
            if model_type in ['LightGBM', 'CatBoost', 'PrototypicalNet']:
                gs = fig.add_gridspec(
                    2, 1, height_ratios=[1.2, 0.8], hspace=0.35,
                    left=0.08, right=0.95, top=0.92, bottom=bottom_margin)
                heatmap_ax = fig.add_subplot(gs[0, 0])
                aux_ax = fig.add_subplot(gs[1, 0])
            else:
                gs = fig.add_gridspec(
                    1, 1, left=0.08, right=0.95, top=0.92, bottom=bottom_margin)
                heatmap_ax = fig.add_subplot(gs[0, 0])
                aux_ax = None

            norm_df = ((numeric_df - numeric_df.min())
                       / (numeric_df.max() - numeric_df.min() + 1e-12))
            display_cols = numeric_cols
            if len(display_cols) > 8:
                top_k = (6 if model_type in ['LightGBM', 'CatBoost', 'PrototypicalNet']
                         else 8)
                var_rank = numeric_df.var().sort_values(ascending=False)
                display_cols = var_rank.head(top_k).index.tolist()
            heatmap_data = norm_df[display_cols].T
            sns.heatmap(heatmap_data, ax=heatmap_ax, cmap='rocket_r',
                        cbar_kws={'label': 'Normalized value'},
                        linewidths=0.4, linecolor='white')
            heatmap_ax.set_title('Hyperparameter landscape', fontsize=12, pad=10)
            heatmap_ax.set_xlabel('Iteration', fontsize=11)
            heatmap_ax.set_ylabel('Hyperparameters', fontsize=11)
            tick_count = min(len(iterations), 15)
            tick_positions = np.linspace(0.5, len(iterations) - 0.5, tick_count)
            tick_indices = [int(np.clip(round(pos - 0.5), 0, len(iterations) - 1))
                            for pos in tick_positions]
            tick_labels = [str(iterations[idx]) for idx in tick_indices]
            heatmap_ax.set_xticks(tick_positions)
            heatmap_ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=9)
            heatmap_ax.tick_params(axis='y', labelsize=9)
            heatmap_ax.axvline(int(np.argmax(scores)) + 0.5, color='white',
                               linestyle='--', linewidth=1.5, alpha=0.9,
                               label='Best iteration')
            heatmap_ax.legend(loc='upper right', fontsize=9, framealpha=0.8)

            if aux_ax is not None:
                top_params_map = {
                    'LightGBM': ['lgb_learning_rate', 'lgb_num_leaves',
                                 'lgb_min_child_samples', 'lgb_max_depth'],
                    'CatBoost': ['cb_learning_rate', 'cb_depth', 'cb_l2_leaf_reg'],
                    'PrototypicalNet': ['lr', 'hidden_dim', 'output_dim', 'epochs'],
                }
                top_params = top_params_map.get(model_type, [])
                available = [p for p in top_params if p in numeric_df.columns]
                if available:
                    colors = plt.cm.Set2(np.linspace(0, 1, len(available)))
                    name_map = {'lr': 'Learning Rate', 'hidden_dim': 'Hidden Dim',
                                'output_dim': 'Output Dim', 'epochs': 'Epochs'}
                    for idx, pname in enumerate(available):
                        clean_name = (name_map.get(pname)
                                      or pname.replace('lgb_', '').replace('cb_', '')
                                      .replace('_', ' ').title())
                        aux_ax.scatter(numeric_df[pname], scores,
                                       label=clean_name, s=60, alpha=0.7,
                                       edgecolor='white', linewidth=0.5,
                                       color=colors[idx])
                    aux_ax.set_xlabel('Parameter Value', fontsize=11)
                    aux_ax.set_ylabel('Accuracy', fontsize=11)
                    aux_ax.set_title('Key Parameter Effects', fontsize=12, pad=10)
                    aux_ax.legend(loc='best', fontsize=9,
                                  ncol=min(2, len(available)), framealpha=0.8)
                    aux_ax.grid(True, alpha=0.3)
                else:
                    aux_ax.axis('off')
    else:
        empty_ax = fig.add_subplot(111)
        empty_ax.text(0.5, 0.5, 'No numeric hyperparameters',
                      ha='center', va='center')
        empty_ax.axis('off')

    plt.tight_layout(rect=[0, bottom_margin, 1, 0.98])
    save_path = os.path.join(save_dir, f'{model_type}_tuning_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()


# ──────────────────────────────────────────────────────────────
#  学习曲线
# ──────────────────────────────────────────────────────────────

def plot_learning_curve(estimator, X, y, model_name,
                        save_dir="results/learning_curves"):
    os.makedirs(save_dir, exist_ok=True)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_mean, color="#1f77b4", marker="o",
             markersize=4, linewidth=2, label="Training accuracy")
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
                     alpha=0.15, color="#1f77b4")
    plt.plot(train_sizes, test_mean, color="#2ca02c", linestyle="--",
             marker="s", markersize=4, linewidth=2, label="Validation accuracy")
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                     alpha=0.15, color="#2ca02c")
    plt.title(f"{model_name} Learning Curve")
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_learning_curve.png"), dpi=300)
    plt.close()


# ──────────────────────────────────────────────────────────────
#  混淆矩阵 + ROC 曲线
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix_and_roc(models, model_names, X_test, y_test, Setname):
    class_labels = ["Control", "ADDwR", "ADDwoR"]

    valid_models = []
    valid_names = []
    for model, name in zip(models, model_names):
        has_predict = hasattr(model, "predict")
        has_forward = hasattr(model, "forward")
        has_predict_proba = hasattr(model, "predict_proba")
        if has_predict or has_forward:
            valid_models.append(model)
            display_name = name.replace('_', ' ').replace('-', ' ')
            valid_names.append(display_name)
            methods = (
                (["predict"] if has_predict else [])
                + (["predict_proba"] if has_predict_proba else [])
                + (["forward"] if has_forward else [])
            )
            print(f"  ✅ {name}: 已添加 (方法: {', '.join(methods)})")
        else:
            print(f"⚠️ 跳过模型 {name}: 没有predict或forward方法")

    if not valid_models:
        print("❌ 没有有效的模型可以绘制")
        return

    print(f"✅ 将绘制 {len(valid_models)} 个模型: {valid_names}")

    # 混淆矩阵
    num_models = len(valid_models)
    cols = 3
    rows = (num_models + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (model, name) in enumerate(zip(valid_models, valid_names)):
        try:
            if hasattr(model, "predict"):
                y_test_pred = model.predict(X_test)
                if isinstance(y_test_pred, np.ndarray) and y_test_pred.ndim == 2:
                    if y_test_pred.shape[1] > 1:
                        y_test_pred = np.argmax(y_test_pred, axis=1)
                    else:
                        y_test_pred = y_test_pred.ravel()
            elif hasattr(model, "forward") and hasattr(model, "eval"):
                import torch
                model.eval()
                with torch.no_grad():
                    X_tensor = (torch.FloatTensor(X_test)
                                if isinstance(X_test, np.ndarray) else X_test)
                    outputs = model(X_tensor)
                    y_test_pred = torch.argmax(outputs, dim=1).numpy()
            else:
                continue
            cm = confusion_matrix(y_test, y_test_pred)
            plt.subplot(rows, cols, i + 1)
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=True,
                        xticklabels=class_labels, yticklabels=class_labels)
            plt.title(f"{name} Confusion Matrix", fontsize=12, fontweight='bold')
            plt.xlabel("Predicted Label", fontsize=10)
            plt.ylabel("True Label", fontsize=10)
        except Exception as e:
            print(f"⚠️ 模型 {name} 绘制混淆矩阵失败: {e}")
            continue

    plt.tight_layout()
    plt.savefig(f'Confusion_Matrix_{Setname}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ROC 曲线
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']

    for class_idx in range(3):
        ax = axes[class_idx]
        for i, (model, name) in enumerate(zip(valid_models, valid_names)):
            try:
                if hasattr(model, 'predict_proba'):
                    try:
                        y_score = model.predict_proba(X_test)
                    except Exception:
                        y_pred = model.predict(X_test)
                        y_score = np.zeros((len(y_pred), len(np.unique(y_test))))
                        y_score[np.arange(len(y_pred)), y_pred] = 1.0
                elif hasattr(model, 'forward') and hasattr(model, 'eval'):
                    import torch
                    model.eval()
                    with torch.no_grad():
                        X_tensor = (torch.FloatTensor(X_test)
                                    if isinstance(X_test, np.ndarray) else X_test)
                        outputs = model(X_tensor)
                        y_score = torch.softmax(outputs, dim=1).numpy()
                else:
                    y_pred_prob = model.predict(X_test)
                    if isinstance(y_pred_prob, np.ndarray) and y_pred_prob.ndim == 2:
                        y_score = y_pred_prob
                    else:
                        continue
                if y_score.ndim == 1:
                    y_score = np.column_stack([1 - y_score, y_score])
                fpr, tpr, _ = roc_curve(y_test_binarized[:, class_idx],
                                        y_score[:, class_idx])
                roc_auc = auc(fpr, tpr)
                color = colors[i % len(colors)]
                ax.plot(fpr, tpr, color=color, lw=2.5,
                        label=f'{name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"⚠️ 跳过模型 {name} 的ROC绘制: {e}")
                continue

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {class_labels[class_idx]}', fontsize=14,
                     fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.6, 0.2, f'Class: {class_labels[class_idx]}',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'ROC_Curve_{Setname}.png', dpi=300, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────
#  t-SNE 可视化
# ──────────────────────────────────────────────────────────────

def visualize_tsne(original_data, lstm_data, labels, titles):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    flattened_original = original_data.reshape(original_data.shape[0], -1)
    tsne = TSNE(n_components=2, random_state=42)
    original_tsne = tsne.fit_transform(flattened_original)
    scatter = plt.scatter(original_tsne[:, 0], original_tsne[:, 1],
                          c=labels, cmap='viridis', alpha=0.6)
    plt.title(titles[0])
    plt.colorbar(scatter)

    plt.subplot(1, 2, 2)
    lstm_tsne = tsne.fit_transform(lstm_data)
    scatter = plt.scatter(lstm_tsne[:, 0], lstm_tsne[:, 1],
                          c=labels, cmap='viridis', alpha=0.6)
    plt.title(titles[1])
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()
