import os

import numpy as np
import pandas as pd
import keras
from imblearn.over_sampling import SMOTE, RandomOverSampler
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from matplotlib import pyplot as plt
from pygments.lexers import objective
from sklearn.metrics import accuracy_score, f1_score, recall_score, auc, roc_curve, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score, KFold, learning_curve
import seaborn as sns
from tensorflow.python.keras.callbacks import ReduceLROnPlateau


def lstm_time_info_extract(X_train_resampled, y_train_resampled, X_val, y_val, X_test):
    # 确保数据类型为 float32
    X_train_resampled = np.array(X_train_resampled, dtype=np.float32)
    y_train_resampled = np.array(y_train_resampled, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    
    print(f'数据类型转换后: X_train_resampled.dtype={X_train_resampled.dtype}')
    
    # 创建并编译 LSTM 模型用于特征提取
    input_shape = (X_train_resampled.shape[1], X_train_resampled.shape[2])
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.4),

        LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(16, return_sequences=False, kernel_regularizer=l2(0.02)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    ])

    # 编译模型以用于特征提取
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['accuracy'])

    # 设置早停和学习率调度器
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, min_delta=0.001)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=5, min_lr=1e-5)

    # 训练 LSTM 模型
    history = model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=64,
        callbacks=[early_stopping, lr_scheduler],
        verbose=2
    )

    # 提取特征
    X_train_features = model.predict(X_train_resampled).reshape(X_train_resampled.shape[0], -1)
    X_test_features = model.predict(X_test).reshape(X_test.shape[0], -1)
    
    print(f'特征提取完成: X_train_features.shape={X_train_features.shape}, dtype={X_train_features.dtype}')
    print(f'特征提取完成: X_test_features.shape={X_test_features.shape}, dtype={X_test_features.dtype}')

    return X_train_features, X_test_features


# 贝叶斯优化函数，增加交叉验证的折数到10折
def bayesian_optimization_cv(model_type, X, y, max_evals=200, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    if model_type == 'SVM':
        space = {
            'C': hp.loguniform('C', np.log(1), np.log(1e2)),  # 调整C参数范围
        }
        gamma_fixed = 'auto'
        kernel_fixed = 'linear'

    elif model_type == 'RandomForest':
        space = {
            'n_estimators': hp.choice('n_estimators', [2, 4, 6, 8, 10]),  # 保证有合理的最小值
            'max_depth': hp.choice('max_depth', [5, 10, 15, 20, 25]),  # 增加树的深度范围
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.05),  # 调整叶节点最小样本比例
            'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.05),  # 增加内部节点划分的最小样本数
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),  # 控制分裂时的最大特征数
            'bootstrap': hp.choice('bootstrap', [True, False])  # 是否使用自举法
        }

    elif model_type == 'DecisionTree':
        space = {
            'max_depth': hp.choice('max_depth', [5, 10, 15, 20, 25]),  # 增加深度选择
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.1),  # 适当调小叶节点最小样本比例
            'criterion': hp.choice('criterion', ['gini', 'entropy']),  # 增加entropy选择
            'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.1),  # 内部节点划分所需的最小样本数
            'splitter': hp.choice('splitter', ['best', 'random'])  # 使用最佳或随机分裂
        }
    else:
        raise ValueError("Unsupported model type")

    def objective(params):
        if model_type == 'SVM':
            model = SVC(C=params['C'], gamma=gamma_fixed, kernel=kernel_fixed, random_state=42, probability=True)
        elif model_type == 'RandomForest':
            # Ensure n_estimators is set to at least 100
            model = RandomForestClassifier(n_estimators=params['n_estimators'],
                                           max_depth=params['max_depth'],
                                           min_samples_leaf=params['min_samples_leaf'],
                                           min_samples_split=params['min_samples_split'],
                                           max_features=params['max_features'],
                                           bootstrap=params['bootstrap'],
                                           random_state=42)

        elif model_type == 'DecisionTree':
            model = DecisionTreeClassifier(max_depth=params['max_depth'],
                                           min_samples_leaf=params['min_samples_leaf'],
                                           min_samples_split=params['min_samples_split'],
                                           criterion=params['criterion'],
                                           splitter=params['splitter'], random_state=42)

        val_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        return {'loss': -np.mean(val_scores), 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best_params


# 函数：增加数据集的样本数量，基于指定的扩展倍数
def random_oversample_for_small_dataset(X_train, y_train, expand_factor):
    """
    X_train: 输入的特征数据
    y_train: 输入的标签数据
    expand_factor: 扩展倍数，例如 1.5 表示样本扩充为原来的 1.5 倍
    """
    # 通过 oversample 扩展样本数量
    ros = SMOTE(sampling_strategy={0: 140, 1: 70, 2: 70}, random_state=42)

    # 整数部分和小数部分分开处理
    int_expand = int(np.floor(expand_factor))  # 整数部分
    frac_expand = expand_factor - int_expand  # 小数部分

    # 处理整数倍扩展
    for _ in range(int_expand - 1):  # int_expand 扩展整数部分
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        X_train = np.concatenate((X_train, X_resampled), axis=0)
        y_train = np.concatenate((y_train, y_resampled), axis=0)

    # 处理小数部分扩展
    if frac_expand > 0:
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        sample_size = int(X_train.shape[0] * frac_expand)

        # 对于小数部分，随机抽取部分样本
        random_indices = np.random.choice(X_resampled.shape[0], sample_size, replace=False)
        X_train = np.concatenate((X_train, X_resampled[random_indices]), axis=0)
        y_train = np.concatenate((y_train, y_resampled[random_indices]), axis=0)

    return X_train, y_train


# 训练模型的函数，加入随机重采样并提供可调的采样策略
def train_models_with_best_params(X_train_features, y_train, weight):
    # 通过 random_oversample_for_small_dataset 函数扩展数据集
    '''X_train_features, y_train = random_oversample_for_small_dataset(X_train_features, y_train, expand_factor=1)
    print(X_train_features.shape)'''

    best_params_svm = bayesian_optimization_cv('SVM', X_train_features, y_train)
    best_params_rf = bayesian_optimization_cv('RandomForest', X_train_features, y_train)
    best_params_dt = bayesian_optimization_cv('DecisionTree', X_train_features, y_train)

    print("Best SVM Hyperparameters:", best_params_svm)
    print("Best RandomForest Hyperparameters:", best_params_rf)
    print("Best DecisionTree Hyperparameters:", best_params_dt)

    # 映射随机森林的参数
    n_estimators_options_rf = [2, 4, 6, 8, 10]
    best_params_rf['n_estimators'] = n_estimators_options_rf[best_params_rf['n_estimators']]  # 映射 n_estimators 的索引
    depth_options_rf = [5, 10, 15, 20, 25]
    best_params_rf['max_depth'] = depth_options_rf[best_params_rf['max_depth']]  # 映射 max_depth 的索引

    max_features_mapping_rf = ['sqrt', 'log2', None]
    best_params_rf['max_features'] = max_features_mapping_rf[best_params_rf['max_features']]  # 映射 max_features
    bootstrap_mapping_rf = [True, False]
    best_params_rf['bootstrap'] = bootstrap_mapping_rf[best_params_rf['bootstrap']]  # 映射 bootstrap

    # 映射决策树的参数
    criterion_mapping_dt = ['gini', 'entropy']
    best_params_dt['criterion'] = criterion_mapping_dt[best_params_dt['criterion']]  # 映射 criterion
    depth_options_dt = [5, 10, 15, 20, 25]
    best_params_dt['max_depth'] = depth_options_dt[best_params_dt['max_depth']]  # 映射 max_depth 的索引
    splitter_mapping_dt = ['best', 'random']
    best_params_dt['splitter'] = splitter_mapping_dt[best_params_dt['splitter']]  # 映射 splitter

    svm_model = SVC(
        C=best_params_svm['C'],
        probability=True,
        random_state=42, class_weight=weight
    )

    # 定义随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=best_params_rf['n_estimators'],  # 映射后的 n_estimators
        max_depth=best_params_rf['max_depth'],  # 映射后的 max_depth
        min_samples_leaf=best_params_rf['min_samples_leaf'],  # min_samples_leaf 直接传递
        min_samples_split=best_params_rf['min_samples_split'],  # 映射后的 min_samples_split
        max_features=best_params_rf['max_features'],  # 映射后的 max_features
        bootstrap=best_params_rf['bootstrap'],  # 映射后的 bootstrap
        random_state=42, class_weight=weight
    )

    # 定义决策树模型
    dt_model = DecisionTreeClassifier(
        max_depth=best_params_dt['max_depth'],  # 映射后的 max_depth
        min_samples_leaf=best_params_dt['min_samples_leaf'],  # min_samples_leaf 直接传递
        min_samples_split=best_params_dt['min_samples_split'],  # 映射后的 min_samples_split
        criterion=best_params_dt['criterion'],  # 映射后的 criterion
        splitter=best_params_dt['splitter'],  # 映射后的 splitter
        random_state=42, class_weight=weight
    )

    # 训练模型
    svm_model.fit(X_train_features, y_train)
    rf_model.fit(X_train_features, y_train)
    dt_model.fit(X_train_features, y_train)

    plot_learning_curve(svm_model, X_train_features, y_train, 'SVM')
    plot_learning_curve(rf_model, X_train_features, y_train, 'RandomForest')
    plot_learning_curve(dt_model, X_train_features, y_train, 'DecisionTree')

    return svm_model, rf_model, dt_model


def plot_learning_curve(estimator, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color="blue", marker="o", markersize=5, label="Training accuracy")
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")
    plt.plot(train_sizes, test_mean, color="green", linestyle="--", marker="s", markersize=5,
             label="Validation accuracy")
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green")
    plt.title(f"Learning Curve: {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def evaluate_model(model, X_train_features, y_train, X_test_features, y_test, model_name, file_info):
    # Training predictions
    y_train_pred = model.predict(X_train_features)
    y_train_prob = model.predict_proba(X_train_features) if hasattr(model, 'predict_proba') else None

    # Testing predictions
    y_test_pred = model.predict(X_test_features)
    y_test_prob = model.predict_proba(X_test_features) if hasattr(model, 'predict_proba') else None

    # Compute metrics for training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_auc = roc_auc_score(y_train, y_train_prob, multi_class="ovr") if y_train_prob is not None else None

    # Compute metrics for test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_auc = roc_auc_score(y_test, y_test_prob, multi_class="ovr") if y_test_prob is not None else None

    print(f"\nModel: {model_name}")
    print(
        f"Training Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}, Recall: {train_recall:.4f}, AUC: {train_auc:.4f}")
    print(
        f"Testing Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}, Recall: {test_recall:.4f}, AUC: {test_auc:.4f}")

    y_result = np.concatenate((y_test, y_train))
    y_result_pred = np.concatenate((y_test_pred, y_train_pred))
    # save_predictions_to_excel(model_name, X_test_features, y_result, y_result_pred, file_info)

    return {
        'train': {'accuracy': train_accuracy, 'f1_score': train_f1, 'recall': train_recall, 'auc': train_auc},
        'test': {'accuracy': test_accuracy, 'f1_score': test_f1, 'recall': test_recall, 'auc': test_auc}
    }


def save_predictions_to_excel(model_name, X_test_features, y_test, y_test_pred, file_info):
    data = []

    # Iterate through the provided file_info list to save predictions
    for idx, (file_name, sheet_index) in enumerate(file_info):
        # You can use X_test_features and y_test_pred to get the data and results
        if idx< len(y_test):
            true_label = y_test[idx]  # Get the true label from y_test
            pred_label = y_test_pred[idx]  # Get the predicted label from y_test_pred

        # Save the prediction along with file name and sheet index
        data.append([file_name, sheet_index, true_label, pred_label])

    # Create DataFrame to save the results
    result_df = pd.DataFrame(data, columns=['File Name', 'Sheet Index', 'True Label', 'Predicted Label'])

    # Save to Excel
    output_path = f'predictions.xlsx'
    with pd.ExcelWriter(output_path, mode='a') as writer:
        result_df.to_excel(writer, sheet_name=model_name, index=False)

    print(f"Predictions saved to {output_path}")


def plot_confusion_matrix_and_roc(models, model_names, X_test, y_test, Setname):
    # 定义类别标签
    class_labels = ["Normal", "DDwR", "DDwoR"]

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 4))
    for i, (model, name) in enumerate(zip(models, model_names)):
        y_test_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_test_pred)

        plt.subplot(1, len(models), i + 1)
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels)  # 设置标签为类别名称
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

    plt.tight_layout()
    plt.savefig(f'Confusion_Matrix_{Setname}.png')
    plt.show()

    # 绘制每个类别的ROC曲线
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])  # 假定类别编码为0, 1, 2
    plt.figure(figsize=(8, 6))
    for i, (model, name) in enumerate(zip(models, model_names)):
        if hasattr(model, 'predict_proba'):
            y_test_prob = model.predict_proba(X_test)
            for j in range(3):  # 针对三分类的每个类别
                fpr, tpr, _ = roc_curve(y_test_binarized[:, j], y_test_prob[:, j])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} - {class_labels[j]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'ROC_Curve_{Setname}.png')
    plt.show()


def summarize_results(results_dict):
    """
    汇总多个模型的评估结果。

    参数:
        results_dict: 字典形式，键为模型名称，值为 evaluate_model() 的返回结果

    示例:
        results_dict = {
            "SVM": result_svm,
            "RandomForest": result_rf,
            "DecisionTree": result_dt
        }
    """
    rows = []
    for model_name, result in results_dict.items():
        row = {
            "Model": model_name,
            "Train Acc": result["train"]["accuracy"],
            "Train Recall": result["train"]["recall"],
            "Train F1": result["train"]["f1_score"],
            "Train AUC": result["train"]["auc"],
            "Test Acc": result["test"]["accuracy"],
            "Test Recall": result["test"]["recall"],
            "Test F1": result["test"]["f1_score"],
            "Test AUC": result["test"]["auc"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("📊 模型性能汇总:")
    print(df.round(4).to_string(index=False))
    return df

