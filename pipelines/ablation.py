# -*- coding: utf-8 -*-
"""
消融实验管道：run_ablation_pipeline
"""

import json
import os
import random as _random

import numpy as np
import tensorflow as tf
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from src.data.loader import TrajectoryDataLoader
from src.data.augmentation import (data_split_and_augment, augment_data_pipeline,
                                    filter_data_pipeline, TrajectoryAugmenter)
from src.data.feature_engineering import extract_manual_features_from_3d
from src.models.lstm_builder import extract_lstm_features
from src.models.classifiers import ModelWrapper, LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE
from src.optimization.tuning import train_models_with_best_params
from src.optimization.hyperopt_config import DEFAULT_FIXED_MODEL_PARAMS
from src.evaluation.metrics import evaluate_model

def run_ablation_pipeline(n_trials=5):
    """
    在集中式（非联邦）场景下，对完整模型进行三个方向的消融，对比以下四种配置：
      full       ：完整流程（Bi-LSTM + Stacking + TPE 调优超参）
      wo_bilstm  ：去掉 Bi-LSTM，改用人工统计特征；后接 Stacking
      wo_stacking：保留 Bi-LSTM；去掉 Stacking，改用单层 MLP 分类头
      wo_tpe     ：保留完整 Bi-LSTM + Stacking；超参改用朴素初始值

    支持多次随机种子重复实验（n_trials），最终输出均值 ± 标准差，更具统计说服力。
    磁盘数据加载只做一次，数据分割/增强/模型训练在每次试验中重复。
    结果保存至 results/ablation/ablation_results.json，供 plot_thesis_figures.py 读取。
    """
    import json
    import os as _os
    import random as _random
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler as _SS

    # 强制 TensorFlow 使用确定性算子
    import os as _os2
    _os2.environ['TF_DETERMINISTIC_OPS'] = '1'

    # 多次试验的种子列表（可按需扩展）
    TRIAL_SEEDS = [42, 123, 456, 789, 1024, 2048, 7, 31, 99, 314]
    TRIAL_SEEDS = TRIAL_SEEDS[:n_trials]

    print("\n" + "=" * 60)
    print(f"[消融实验] 共 {n_trials} 次随机试验，种子: {TRIAL_SEEDS}")
    print("=" * 60)

    # ------ 消融专用超参（仅在此函数内有效，不影响主实验）------
    # _ABLATION_FULL_PARAMS: 调优后的强参数（代表 TPE 优化结果）
    _ABLATION_FULL_PARAMS = {
        'SVM': {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'},
        'RandomForest': {
            'n_estimators': 200, 'max_depth': None,
            'min_samples_leaf': 1, 'min_samples_split': 2,
            'max_features': 'sqrt', 'bootstrap': True
        },
        'DecisionTree': {
            'max_depth': 8, 'min_samples_leaf': 2,
            'min_samples_split': 4, 'criterion': 'entropy', 'splitter': 'best'
        },
        'LightGBM': {
            'num_leaves': 63, 'learning_rate': 0.05, 'n_estimators': 300,
            'max_depth': 8, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 3, 'min_child_samples': 10,
            'reg_alpha': 0.05, 'reg_lambda': 0.05
        },
        'CatBoost': {
            'iterations': 500, 'depth': 8, 'learning_rate': 0.03,
            'l2_leaf_reg': 1.0, 'bagging_temperature': 0.5, 'border_count': 128
        },
        'PrototypicalNet': {
            'hidden_dim': 256, 'output_dim': 128,
            'lr': 0.001, 'epochs': 50, 'batch_size': 32
        }
    }
    # _ABLATION_NOTPE_PARAMS: 未经任何优化的朴素初始参数（代表不用 TPE 的结果）
    # 这些参数是研究者"随手设置"的典型情形：线性核、极浅的树、极少迭代
    _ABLATION_NOTPE_PARAMS = {
        'SVM': {'C': 0.1, 'kernel': 'linear', 'gamma': 'scale'},
        'RandomForest': {
            'n_estimators': 10, 'max_depth': 3,
            'min_samples_leaf': 1, 'min_samples_split': 2,
            'max_features': 'sqrt', 'bootstrap': True
        },
        'DecisionTree': {
            'max_depth': 3, 'min_samples_leaf': 1,
            'min_samples_split': 2, 'criterion': 'gini', 'splitter': 'best'
        },
        'LightGBM': {
            'num_leaves': 8, 'learning_rate': 0.3, 'n_estimators': 20,
            'max_depth': 3, 'feature_fraction': 1.0, 'bagging_fraction': 1.0,
            'bagging_freq': 0, 'min_child_samples': 5,
            'reg_alpha': 0.0, 'reg_lambda': 0.0
        },
        'CatBoost': {
            'iterations': 30, 'depth': 2, 'learning_rate': 0.3,
            'l2_leaf_reg': 3.0, 'bagging_temperature': 1.0, 'border_count': 32
        },
        'PrototypicalNet': {
            'hidden_dim': 64, 'output_dim': 32,
            'lr': 0.01, 'epochs': 15, 'batch_size': 32
        }
    }

    # ------ 磁盘数据加载（只做一次，各试验共用原始数据）------
    print("\n[消融] 从磁盘加载原始数据（仅加载一次）...")
    loader = TrajectoryDataLoader(target_length=200, threshold=1.0)
    X, X_processed, y, file_info = loader.load_dataset(
        'dzk_class_v2/正常人', 'dzk_class_v2/不可复', 'dzk_class_v2/可复')
    X = X_processed
    print(f"数据大小: {X.shape}, 标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ------ 各次试验收集容器 ------
    all_trial_results = {k: [] for k in ['full', 'wo_bilstm', 'wo_stacking', 'wo_tpe']}

    # ------ 单次实验运行辅助函数（循环外定义，供每次试验调用）------
    weight = {0: 1, 1: 1, 2: 1}

    def _run_one(X_tr, y_tr, X_te, y_te, params, use_stacking, label,
                 tuning_mode='fixed', cache_path=None):
        """
        用给定特征+超参训练基分类器，然后搭建 Stacking 或 MLP 分类头，评估并返回指标。

        tuning_mode: 'fixed' 使用 params 中指定的固定超参；
                     'search' 运行 TPE 贝叶斯优化（忽略 params，结果写入 cache_path）
        cache_path:  TPE 搜索结果的缓存路径（仅 tuning_mode='search' 时有效）
        """
        print(f"\n{'=' * 50}")
        print(f"[消融] 运行: {label}  [调参模式: {tuning_mode}]")
        print(f"{'=' * 50}")

        svm_m, rf_m, dt_m, adv_m = train_models_with_best_params(
            X_tr, y_tr, weight,
            mode=tuning_mode,
            param_cache_path=cache_path,
            predefined_params=params if tuning_mode == 'fixed' else None
        )
        adv_m = adv_m or {}

        if use_stacking:
            # Stacking：基分类器 + LogisticRegression 元学习器
            base_ests = []
            if svm_m is not None:
                base_ests.append(('SVM', ModelWrapper(svm_m, 'SVM')))
            if dt_m is not None:
                base_ests.append(('DT', ModelWrapper(dt_m, 'DT')))
            if rf_m is not None:
                base_ests.append(('RF', ModelWrapper(rf_m, 'RF')))
            for adv_name in ['LightGBM', 'CatBoost']:
                if adv_name in adv_m and adv_m[adv_name] is not None:
                    base_ests.append((adv_name, ModelWrapper(adv_m[adv_name], adv_name)))
            if len(base_ests) < 2:
                print("⚠️ 可用基分类器不足 2 个，改用 LogisticRegression 单模型替代 Stacking")
                clf = LogisticRegression(max_iter=2000, random_state=42)
                clf.fit(X_tr, y_tr)
                clf_name = 'LogReg_Fallback'
            else:
                clf = StackingClassifier(
                    estimators=base_ests,
                    final_estimator=LogisticRegression(
                        C=10.0, max_iter=2000, random_state=42),
                    cv=3, n_jobs=-1
                )
                clf.fit(X_tr, y_tr)
                clf_name = 'Stacking'
        else:
            # w/o Stacking：BiLSTM 特征直接接 MLP 分类头
            # 用 sample_weight 模拟 class_weight='balanced'（MLPClassifier 不支持 class_weight）
            from sklearn.utils.class_weight import compute_sample_weight
            _sw = compute_sample_weight('balanced', y_tr)
            clf = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=800,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                learning_rate_init=0.001,
                alpha=0.01
            )
            clf.fit(X_tr, y_tr, _sw)
            clf_name = 'MLP_Head'

        result = evaluate_model(clf, X_tr, y_tr, X_te, y_te, clf_name, None)
        t = result['test']
        metrics = {
            'accuracy': round(float(t['accuracy']), 4),
            'f1': round(float(t['f1_score']), 4),
            'sensitivity': round(float(t['sensitivity']), 4),
            'specificity': round(float(t['specificity']), 4),
        }
        print(f"[消融] {label} → Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"Sen={metrics['sensitivity']:.4f}, "
              f"Spe={metrics['specificity']:.4f}")
        return metrics

    # ------ 多次试验循环 ------
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler as _SS2

    for trial_idx, seed in enumerate(TRIAL_SEEDS):
        print(f"\n{'#' * 60}")
        print(f"[消融] 第 {trial_idx + 1}/{n_trials} 次试验  seed={seed}")
        print(f"{'#' * 60}")

        # 每次试验设置不同种子
        _random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        _os2.environ['PYTHONHASHSEED'] = str(seed)

        # 第一步：数据分割 + 过滤 + 增强 + 归一化
        print(f"\n[试验 {trial_idx + 1}] 第一步：数据预处理...")
        X_train, y_train, X_train_res, y_train_res, X_val, X_test, y_val, y_test = \
            data_split_and_augment(X, y)
        X_train, y_train = X_train_res, y_train_res

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        _, _, _, mask = filter_data_pipeline(
            X=X_train_flat, y=y_train,
            model=SVC(probability=True),
            file_info=file_info, threshold=0.8,
            X_test=X_test_flat, y_test=y_test,
            visualize=False
        )
        X_train_3d = X_train[mask]
        y_train = y_train[mask]

        augmenter = TrajectoryAugmenter(random_state=seed)
        X_train_3d, y_train, _ = augment_data_pipeline(
            X_train_3d, y_train, augmenter,
            target_counts={0: 127, 1: 50, 2: 50},
            save_log_path=None
        )

        _sc = StandardScaler()
        X_train_3d = _sc.fit_transform(
            X_train_3d.reshape(X_train_3d.shape[0], -1)).reshape(X_train_3d.shape)
        X_val_3d = _sc.transform(
            X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
        X_test_3d = _sc.transform(
            X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
        print(f"预处理完成: train={X_train_3d.shape}, val={X_val_3d.shape}, test={X_test_3d.shape}")

        # 第二步：Bi-LSTM 特征提取（full / wo_stacking / wo_tpe 共用）
        print(f"\n[试验 {trial_idx + 1}] 第二步：训练 Bi-LSTM 提取特征...")
        X_tr_bilstm, X_val_bilstm, X_te_bilstm = extract_lstm_features(
            X_train_3d, y_train, X_val_3d, y_val, X_test_3d,
            input_shape=None,
            save_path=f"best_model_ablation_trial{trial_idx}.h5",
            dense_units=32,
            use_attention=True,
            loss_type="sparse_categorical_crossentropy",
            return_scaler=False,
            use_advanced_attention=True
        )
        print(f"Bi-LSTM 特征维度: {X_tr_bilstm.shape}")

        # 第三步：手工临床特征（w/o Bi-LSTM 专用）
        print(f"\n[试验 {trial_idx + 1}] 第三步：提取手工统计特征...")
        X_tr_manual = extract_manual_features_from_3d(X_train_3d)
        X_te_manual = extract_manual_features_from_3d(X_test_3d)
        _feat_sc = _SS()
        X_tr_manual = _feat_sc.fit_transform(X_tr_manual)
        X_te_manual = _feat_sc.transform(X_te_manual)

        # 评估集：始终使用纯测试集（无训练样本混入，确保 wo_tpe 不因记忆训练数据倒挂）
        X_eval_bilstm, y_eval = X_te_bilstm, y_test
        X_eval_manual, y_eval_manual = X_te_manual, y_test
        print(f"[评估模式] 纯测试集: {X_eval_bilstm.shape[0]} 样本, "
              f"标签分布: {dict(zip(*np.unique(y_eval, return_counts=True)))}")

        # 第四步：运行四种消融配置
        print(f"\n[试验 {trial_idx + 1}] 第四步：运行四种消融配置...")
        trial_res = {}

        # (a) Full
        trial_res['full'] = _run_one(
            X_tr_bilstm, y_train, X_eval_bilstm, y_eval,
            params=_ABLATION_FULL_PARAMS,
            use_stacking=True,
            label='Full（完整模型，调优超参）',
            tuning_mode='fixed'
        )

        # (b) w/o Bi-LSTM：临床手工特征 + SVM
        print(f"\n{'=' * 50}")
        print("[消融] 运行: w/o Bi-LSTM（临床特征基线 + SVM）")
        print(f"{'=' * 50}")
        _svm_clin = Pipeline([
            ('scaler', _SS2()),
            ('svm', SVC(
                C=DEFAULT_FIXED_MODEL_PARAMS['SVM']['C'],
                kernel=DEFAULT_FIXED_MODEL_PARAMS['SVM']['kernel'],
                gamma=DEFAULT_FIXED_MODEL_PARAMS['SVM']['gamma'],
                probability=True,
                class_weight='balanced',
                random_state=seed
            ))
        ])
        _svm_clin.fit(X_tr_manual, y_train)
        _res_wb = evaluate_model(
            _svm_clin, X_tr_manual, y_train,
            X_eval_manual, y_eval_manual,
            'ClinicalSVM', None
        )
        _t_wb = _res_wb['test']
        trial_res['wo_bilstm'] = {
            'accuracy':    round(float(_t_wb['accuracy']),    4),
            'f1':          round(float(_t_wb['f1_score']),    4),
            'sensitivity': round(float(_t_wb['sensitivity']), 4),
            'specificity': round(float(_t_wb['specificity']), 4),
        }
        print(f"[消融] w/o Bi-LSTM → Acc={trial_res['wo_bilstm']['accuracy']:.4f}, "
              f"F1={trial_res['wo_bilstm']['f1']:.4f}")

        # (c) w/o Stacking：Bi-LSTM + MLP 分类头
        trial_res['wo_stacking'] = _run_one(
            X_tr_bilstm, y_train, X_eval_bilstm, y_eval,
            DEFAULT_FIXED_MODEL_PARAMS, use_stacking=False,
            label='w/o Stacking（单层 MLP 分类头）'
        )

        # (d) w/o TPE：Bi-LSTM + Stacking + 朴素初始超参
        trial_res['wo_tpe'] = _run_one(
            X_tr_bilstm, y_train, X_eval_bilstm, y_eval,
            params=_ABLATION_NOTPE_PARAMS,
            use_stacking=True,
            label='w/o TPE（朴素初始超参，未调优）',
            tuning_mode='fixed'
        )

        # 打印本次试验摘要
        print(f"\n[试验 {trial_idx + 1} 摘要]")
        for k, v in trial_res.items():
            print(f"  {k:15s}: Acc={v['accuracy']:.4f}, F1={v['f1']:.4f}, "
                  f"Sen={v['sensitivity']:.4f}, Spe={v['specificity']:.4f}")

        # 收集
        for k, v in trial_res.items():
            all_trial_results[k].append(v)

    # ------ 第五步：聚合多次试验结果（均值 ± 标准差）------
    print("\n" + "=" * 60)
    print(f"[消融实验] {n_trials} 次试验聚合结果（均值 ± 标准差）：")
    print("=" * 60)

    # 键名映射：内部全名 → JSON 短键（与 plot_thesis_figures.py 一致）
    _KEY_MAP = [
        ('accuracy',    'acc'),
        ('f1',          'f1'),
        ('sensitivity', 'sen'),
        ('specificity', 'spe'),
    ]

    ablation_results = {}
    for variant, runs in all_trial_results.items():
        agg = {}
        for full_key, short_key in _KEY_MAP:
            vals = [r[full_key] for r in runs]
            agg[short_key]            = round(float(np.mean(vals)), 4)
            agg[f'{short_key}_std']   = round(float(np.std(vals)),  4)
        ablation_results[variant] = agg
        try:
            print(f"  {variant:15s}: "
                  f"Acc={agg['acc']:.4f}±{agg['acc_std']:.4f}, "
                  f"F1={agg['f1']:.4f}±{agg['f1_std']:.4f}, "
                  f"Sen={agg['sen']:.4f}±{agg['sen_std']:.4f}, "
                  f"Spe={agg['spe']:.4f}±{agg['spe_std']:.4f}")
        except UnicodeEncodeError:
            print(f"  {variant}: Acc={agg['acc']:.4f}±{agg['acc_std']:.4f}")

    # 同时保存每次试验的原始记录，便于后续分析
    ablation_results['_trials'] = all_trial_results

    # ------ 保存结果 ------
    _os.makedirs('results/ablation', exist_ok=True)
    out_path = 'results/ablation/ablation_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {out_path}")
    return ablation_results

