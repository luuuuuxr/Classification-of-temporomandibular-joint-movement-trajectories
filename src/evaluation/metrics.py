# -*- coding: utf-8 -*-
"""
评估指标模块：
  calculate_sensitivity_specificity / evaluate_model /
  summarize_results / save_results_to_excel
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, recall_score,
                              roc_auc_score)


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


def evaluate_model(model, X_train_features, y_train,
                   X_test_features, y_test, model_name, file_info=None):
    if hasattr(model, "predict_proba"):
        y_train_pred = model.predict(X_train_features)
        y_test_pred = model.predict(X_test_features)
        y_train_prob = model.predict_proba(X_train_features)
        y_test_prob = model.predict_proba(X_test_features)
    else:
        y_train_pred = model.predict(X_train_features)
        y_test_pred = model.predict(X_test_features)
        if isinstance(y_train_pred, np.ndarray) and y_train_pred.ndim > 1:
            y_train_pred = np.argmax(y_train_pred, axis=1)
            y_test_pred = np.argmax(y_test_pred, axis=1)
            y_train_prob = y_train_pred
            y_test_prob = y_test_pred
        else:
            y_train_prob = None
            y_test_prob = None

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    try:
        train_auc = roc_auc_score(y_train, y_train_prob, multi_class="ovr")
    except Exception:
        train_auc = None

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    try:
        test_auc = roc_auc_score(y_test, y_test_prob, multi_class="ovr")
    except Exception:
        test_auc = None

    train_sen, train_spe = calculate_sensitivity_specificity(y_train, y_train_pred)
    test_sen, test_spe = calculate_sensitivity_specificity(y_test, y_test_pred)

    print(f"\n📊 [模型评估] {model_name}")
    print(f"✅ 训练集: Accuracy={train_accuracy:.4f}, F1={train_f1:.4f}, "
          f"Recall={train_recall:.4f}, AUC={train_auc}, "
          f"Sensitivity={train_sen:.4f}, Specificity={train_spe:.4f}")
    print(f"✅ 测试集: Accuracy={test_accuracy:.4f}, F1={test_f1:.4f}, "
          f"Recall={test_recall:.4f}, AUC={test_auc}, "
          f"Sensitivity={test_sen:.4f}, Specificity={test_spe:.4f}")

    errors = y_test != y_test_pred
    print(f"\n❌ 错误预测样本数: {np.sum(errors)} / {len(y_test)}")
    if file_info is not None:
        print("部分错误样本追踪（最多前5个）:")
        for idx in np.where(errors)[0][:5]:
            print(f" - True: {y_test[idx]} | Pred: {y_test_pred[idx]} | "
                  f"File: {file_info[idx]}")

    print("\n📋 分类报告:")
    try:
        target_names = ["Control", "ADDwR", "ADDwoR"]
        print(classification_report(y_test, y_test_pred, target_names=target_names))
    except Exception:
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


def save_results_to_excel(file_name, file_info, sheet_name,
                          y_true, y_pred_svm, y_pred_rf, y_pred_dt):
    results = pd.DataFrame({
        'File Info': file_info,
        'Sheet Name': sheet_name,
        'True Label': y_true,
        'SVM Prediction': y_pred_svm,
        'RandomForest Prediction': y_pred_rf,
        'DecisionTree Prediction': y_pred_dt
    })
    with pd.ExcelWriter(file_name, engine='openpyxl', mode='a') as writer:
        results.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Results saved to {file_name} in sheet: {sheet_name}")
