# -*- coding: utf-8 -*-
"""
标准流程管道：run_standard_pipeline
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.loader import TrajectoryDataLoader
from src.data.augmentation import (data_split_and_augment, augment_data_pipeline,
                                    filter_data_pipeline, TrajectoryAugmenter)
from src.models.lstm_builder import extract_lstm_features
from src.models.classifiers import (LightGBMClassifier, CatBoostClassifier,
                                     PrototypicalNetwork, ModelWrapper,
                                     LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE)
from src.models.enhanced_lstm_model import EnhancedLSTMFeatureExtractor
from src.optimization.tuning import train_models_with_best_params
from src.optimization.hyperopt_config import BEST_PARAM_CACHE_PATH, DEFAULT_FIXED_MODEL_PARAMS
from src.evaluation.metrics import evaluate_model, summarize_results
from src.evaluation.plots import plot_confusion_matrix_and_roc, visualize_tsne
from src.visualization.model_explainability import (
    plot_phase_shap_importance, plot_phase_aggregated_feature_importance,
    plot_ensemble_shap_importance, plot_classwise_phase_shap_analysis,
    plot_spatial_dimension_shap_analysis
)
from src.visualization.improved_explainability import comprehensive_phase_analysis
from src.utils.trajectory_build import split_into_phases, plot_phases_3d

try:
    import lightgbm as lgb
except ImportError:
    pass
try:
    import catboost as cb
except ImportError:
    pass

def run_standard_pipeline():
    # ============ 配置选项 ============
    # 是否使用增强LSTM特征提取 (True: 使用增强LSTM, False: 使用原始LSTM)
    USE_ENHANCED_LSTM = False
    # 模型训练模式：'search' 进行贝叶斯调参，'fixed' 直接使用已知最佳参数
    PARAM_TUNING_MODE = 'fixed'
    
    # ============ 读取数据并初步处理数据 ============
    loader = TrajectoryDataLoader(target_length=200, threshold=1.0)

    X, X_processed, y, file_info = loader.load_dataset(
        'dzk_class_v2/正常人', 'dzk_class_v2/不可复', 'dzk_class_v2/可复')

    # 你可以选择用处理过的：
    X = X_processed

    print("✅ 数据集读取完成！\n")
    print("数据大小：", X.shape)
    print("标签分布：", dict(zip(*np.unique(y, return_counts=True))))

    # ============ 数据集分割及随机重采样 ============
    print("\n🔁 正在进行数据集分割...")
    X_train, y_train, \
        X_train_resampled, y_train_resampled, \
        X_val, X_test, y_val, y_test = data_split_and_augment(X, y)
    y_test_for_shap = y_test.copy()

    X_train = X_train_resampled
    y_train = y_train_resampled
    print("✅ 数据集分割完成！")

    # ========== 🚫 先过滤掉低质量样本 ==========
    # 👇 将 3D 轨迹数据 (samples, 200, 3) 展平为 (samples, 600)
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)

    # 然后用于筛选的是 lstm 特征：
    X_train_filtered, y_train_filtered, removed_samples, mask = filter_data_pipeline(
        X=X_train_flattened,
        y=y_train,
        model=SVC(probability=True),
        file_info=file_info,
        threshold=0.8,
        X_test=X_test_flattened,
        y_test=y_test,
        visualize=True
    )

    # ========== ✅ 然后再进行增强 ==========
    # 从原始3D数据中找到保留的样本
    X_train_3d_filtered = X_train[mask]
    y_train_3d_filtered = y_train[mask]

    # 使用固定随机种子创建增强器，确保结果可复现
    augmenter = TrajectoryAugmenter(random_state=42)
    # Control类固定为127个样本，确保结果可复现
    X_augmented, y_augmented, aug_log = augment_data_pipeline(
        X_train_3d_filtered, y_train_3d_filtered, augmenter,
        target_counts={0: 127, 1: 50, 2: 50}, save_log_path='augmentation_log.xlsx'
    )

    # =============== 数据归一化阶段 ================
    scaler = StandardScaler()
    X_train = X_augmented
    y_train = y_augmented
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
    X_test_scaler = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    # ============ 特征提取 ============
    if USE_ENHANCED_LSTM:
        print("🚀 使用增强LSTM进行特征提取...")
        # 使用增强LSTM特征提取
        extractor = EnhancedLSTMFeatureExtractor(
            lstm_units=[128, 64],
            attention_heads=16,
            dense_units=128,
            dropout_rate=0.5
        )
        
        # 训练特征提取器
        print("�� 训练增强LSTM特征提取器...")
        history = extractor.train_feature_extractor(
            X_train, y_train, X_val, y_val,
            epochs=100,
            batch_size=64,
            save_path="enhanced_lstm_model.h5"
        )
        
        # 提取特征
        print("🔍 提取训练集、验证集和测试集特征...")
        # extract_features_with_scaling 返回 (train_features_scaled, test_features_scaled)
        # 我们需要分别提取并标准化
        train_features_raw = extractor.extract_features(X_train)
        val_features_raw = extractor.extract_features(X_val)
        test_features_raw = extractor.extract_features(X_test)
        
        # 使用训练集拟合scaler，然后transform所有特征
        # StandardScaler已在文件开头导入
        scaler = StandardScaler()
        X_train_features = scaler.fit_transform(train_features_raw)
        X_val_features = scaler.transform(val_features_raw)
        X_test_features = scaler.transform(test_features_raw)
        
        print(f"✅ 增强LSTM特征提取完成!")
        print(f"训练集特征形状: {X_train_features.shape}")
        print(f"验证集特征形状: {X_val_features.shape}")
        print(f"测试集特征形状: {X_test_features.shape}")
    else:
        print("🔍 使用原始LSTM进行特征提取...")
        X_train_features, X_val_features, X_test_features = extract_lstm_features(X_train, y_train, X_val, y_val, X_test_scaler,
                                                                  input_shape=None,
                                                                  save_path="best_model.h5",
                                                                  dense_units=32,  # ✅ 输出维度
                                                                  use_attention=True,  # ✅ 是否使用 Attention
                                                                  loss_type="sparse_categorical_crossentropy",
                                                                  # ✅ 损失函数类型
                                                                  return_scaler=False,  # ✅ 是否返回 scaler
                                                                  use_advanced_attention=True  # ✅ 启用增强注意力机制
                                                                  )

    print("训练集在经过LSTM提取时序信息后大小：")
    print(X_train_features.shape)
    print("验证集在经过LSTM提取时序信息后大小：")
    print(X_val_features.shape)
    print("测试集在经过LSTM提取时序信息后大小：")
    print(X_test_features.shape)

    visual_tsne = True
    if visual_tsne:
        visualize_tsne(
            X_train,
            X_train_features,
            y_train,
            ["t-SNE of Raw Trajectory Data", "t-SNE of LSTM-processed Features"]
        )

    weight = {0: 1, 1: 1, 2: 1}
    svm_model, rf_model, dt_model, advanced_models = train_models_with_best_params(
        X_train_features,
        y_train,
        weight,
        mode=PARAM_TUNING_MODE,
        param_cache_path=BEST_PARAM_CACHE_PATH,
        predefined_params=DEFAULT_FIXED_MODEL_PARAMS
    )
    advanced_models = advanced_models or {}

    # ================= 模型评估阶段 ===============
    result_svm = evaluate_model(svm_model, X_train_features, y_train, X_test_features, y_test, 'SVM', file_info)
    result_rf = evaluate_model(rf_model, X_train_features, y_train, X_test_features, y_test, 'RandomForest', file_info)
    result_dt = evaluate_model(dt_model, X_train_features, y_train, X_test_features, y_test, 'DecisionTree', file_info)
    
    # 初始化结果字典
    results_dict = {
        'SVM': result_svm,
        'RandomForest': result_rf,
        'DecisionTree': result_dt,
    }
    # ================= 先进分类器测试阶段 ===============
    print('\n' + '=' * 60)
    print('🚀 开始测试最新先进分类器...')
    print('=' * 60)
    
    # 准备数据（展平为2D特征）
    X_train_flat = X_train_features
    X_test_flat = X_test_features
    
    # 测试先进分类器
    print(f'输入数据形状: X_train_flat: {X_train_flat.shape}, X_test_flat: {X_test_flat.shape}')
    print(f'标签形状: y_train: {y_train.shape}, y_test: {y_test.shape}')
    
    try:
        input_dim = X_train_flat.shape[1]
        if not advanced_models:
            print('ℹ️ 未检测到已训练的先进模型，开始快速训练默认版本...')
            advanced_models = {}

            # 1. LightGBM
            try:
                if LIGHTGBM_AVAILABLE:
                    print('🔍 训练 LightGBM...')
                    lgb_model = LightGBMClassifier()
                    lgb_model.fit(X_train_flat, y_train)
                    advanced_models['LightGBM'] = lgb_model
                    print('✅ LightGBM 训练成功')
                else:
                    print('⚠️ LightGBM 不可用')
            except Exception as e:
                print(f'❌ LightGBM 训练失败: {e}')

            # 2. CatBoost
            try:
                if CATBOOST_AVAILABLE:
                    print('🔍 训练 CatBoost...')
                    cat_model = CatBoostClassifier()
                    cat_model.fit(X_train_flat, y_train)
                    advanced_models['CatBoost'] = cat_model
                    print('✅ CatBoost 训练成功')
                else:
                    print('⚠️ CatBoost 不可用')
            except Exception as e:
                print(f'❌ CatBoost 训练失败: {e}')

            # 3. Prototypical Network
            try:
                print('🔍 训练 Prototypical Network...')
                proto_model = PrototypicalNetwork(input_dim=input_dim)
                proto_model.fit(X_train_flat, y_train, epochs=50)
                advanced_models['PrototypicalNet'] = proto_model
                print('✅ Prototypical Network 训练成功')
            except Exception as e:
                print(f'❌ Prototypical Network 训练失败: {e}')
        else:
            print('✅ 使用已调参的先进模型，跳过重复训练')

        # 将先进分类器结果添加到结果字典
        advanced_model_names = ['LightGBM', 'CatBoost', 'PrototypicalNet']

        for name in advanced_model_names:
            if name in advanced_models:
                model = advanced_models[name]

                # 使用evaluate_model函数评估模型
                result = evaluate_model(model, X_train_flat, y_train, X_test_flat, y_test, name,
                                        file_info)

                # 添加到结果字典
                results_dict[name] = result

        print(f'\n✅ 先进分类器测试完成！')
        
    except Exception as e:
        print(f'⚠️ 先进分类器测试失败: {e}')
        print('继续使用传统分类器...')
        # 确保 results_dict 仍然可用

    # ================= 自适应加权集成学习框架 ===============
    print('\n' + '=' * 60)
    print('🎯 构建集成学习框架')
    print('=' * 60)
    
    # ============ 智能模型选择策略 ============
    def select_ensemble_models(base_models_dict, advanced_models_dict=None, strategy='all'):
        """
        智能选择集成学习的模型组合
        
        参数:
            base_models_dict: 基础模型字典 {'SVM': model, ...}
            advanced_models_dict: 先进模型字典 {'LightGBM': model, ...}
            strategy: 选择策略
                - 'all': 使用所有模型（推荐用于最终实验）
                - 'diverse': 优先多样性（推荐用于论文）
                - 'best': 只选性能最好的模型
                - 'balanced': 性能与多样性平衡
        
        返回:
            models: 模型列表
            model_names: 模型名称列表
        """
        models = []
        model_names = []
        
        # 定义模型类型和多样性分数
        model_diversity = {
            'SVM': {'type': 'kernel', 'diversity': 3, 'priority': 1},
            'RandomForest': {'type': 'tree_ensemble', 'diversity': 3, 'priority': 1},
            'DecisionTree': {'type': 'tree', 'diversity': 2, 'priority': 3},
            'LightGBM': {'type': 'gradient_boosting', 'diversity': 3, 'priority': 1},
            'CatBoost': {'type': 'gradient_boosting', 'diversity': 2, 'priority': 2},
            'PrototypicalNet': {'type': 'metric_learning', 'diversity': 3, 'priority': 1}
        }
        
        if strategy == 'all':
            # 使用所有可用模型
            print('\n📌 策略: 使用所有可用模型（最大化集成效果）')
            for name, model in base_models_dict.items():
                models.append(model)
                model_names.append(name)
            if advanced_models_dict:
                for name, model in advanced_models_dict.items():
                    models.append(model)
                    model_names.append(name)
        
        elif strategy == 'diverse':
            # 优先选择多样性高的模型
            print('\n📌 策略: 优先多样性（推荐用于论文展示）')
            priority_models = ['SVM', 'RandomForest', 'LightGBM', 'PrototypicalNet']
            
            for name in priority_models:
                if name in base_models_dict:
                    models.append(base_models_dict[name])
                    model_names.append(name)
                elif advanced_models_dict and name in advanced_models_dict:
                    models.append(advanced_models_dict[name])
                    model_names.append(name)
        
        elif strategy == 'best':
            # 只选性能最好的模型
            print('\n📌 策略: 只选最佳模型（保守策略）')
            priority_models = ['LightGBM', 'CatBoost', 'RandomForest', 'SVM']
            count = 0
            for name in priority_models:
                if count >= 4:
                    break
                if name in base_models_dict:
                    models.append(base_models_dict[name])
                    model_names.append(name)
                    count += 1
                elif advanced_models_dict and name in advanced_models_dict:
                    models.append(advanced_models_dict[name])
                    model_names.append(name)
                    count += 1
        
        elif strategy == 'balanced':
            # 性能与多样性平衡
            print('\n📌 策略: 性能与多样性平衡')
            balanced_models = ['SVM', 'RandomForest', 'LightGBM', 'CatBoost']
            
            for name in balanced_models:
                if name in base_models_dict:
                    models.append(base_models_dict[name])
                    model_names.append(name)
                elif advanced_models_dict and name in advanced_models_dict:
                    models.append(advanced_models_dict[name])
                    model_names.append(name)
        
        return models, model_names
    
    # 准备模型字典
    base_models_dict = {
        'SVM': svm_model,
        'RandomForest': rf_model,
        'DecisionTree': dt_model
    }
    
    # 获取先进分类器
    advanced_models_dict = None
    if 'advanced_models' in locals() and advanced_models:
        advanced_models_dict = advanced_models
    
    # ============ 准备验证集和测试集标签 ============
    # 注意：X_val_features 已经在特征提取阶段生成
    # 使用原始的验证集和测试集标签
    y_val_final = y_val  # 使用原始验证集标签
    y_test_final = y_test  # 使用原始测试集标签
    X_test_final_features = X_test_features  # 使用全部测试集作为最终测试集
    
    # ============ 智能模型选择：评估并过滤弱模型 ============
    print('\n📋 智能选择集成学习模型')
    print('=' * 60)
    
    # 步骤1: 评估所有模型在验证集上的性能
    print('\n🔍 步骤1: 评估所有模型性能...')
    model_performance = {}
    
    # 评估传统模型
    for name, model in [('SVM', svm_model), ('RandomForest', rf_model), ('DecisionTree', dt_model)]:
        try:
            if hasattr(model, 'score'):
                acc = model.score(X_val_features, y_val_final)
            else:
                pred = model.predict(X_val_features)
                acc = (pred == y_val_final).mean()
            model_performance[name] = acc
            print(f'  {name:15s}: {acc:.4f}')
        except Exception as e:
            print(f'  {name:15s}: 评估失败 ({e})')
            model_performance[name] = 0.0
    
    # 评估先进模型
    if 'advanced_models' in locals() and advanced_models:
        for name in ['LightGBM', 'CatBoost', 'PrototypicalNet']:
            if name in advanced_models:
                try:
                    model = advanced_models[name]
                    if hasattr(model, 'score'):
                        acc = model.score(X_val_features, y_val_final)
                    elif hasattr(model, 'predict'):
                        pred = model.predict(X_val_features)
                        acc = (pred == y_val_final).mean()
                    else:
                        import torch
                        logits = model.forward(torch.FloatTensor(X_val_features))
                        pred = logits.argmax(dim=1).detach().numpy()
                        acc = (pred == y_val_final).mean()
                    model_performance[name] = acc
                    print(f'  {name:15s}: {acc:.4f}')
                except Exception as e:
                    print(f'  {name:15s}: 评估失败 ({e})')
                    model_performance[name] = 0.0
    
    # 步骤2: 智能选择模型（过滤弱模型用于Stacking，但保留所有模型用于绘图）
    print('\n🎯 步骤2: 选择模型组合（用于Stacking训练）...')
    
    # 首先保存所有模型（用于绘图）- 直接从原始变量获取，确保包含所有模型
    all_models_for_plot = []
    all_model_names_for_plot = []
    
    # 添加所有传统模型
    for name, model in [('SVM', svm_model), ('RandomForest', rf_model), ('DecisionTree', dt_model)]:
        if model is not None:
            all_models_for_plot.append(model)
            all_model_names_for_plot.append(name)
    
    # 添加所有先进模型 - 直接从advanced_models获取，如果不存在则尝试从locals获取
    print(f'   🔍 检查advanced_models: {"存在" if "advanced_models" in locals() else "不存在"}')
    if 'advanced_models' in locals() and advanced_models:
        print(f'   📋 advanced_models包含的键: {list(advanced_models.keys())}')
        # 优先从advanced_models字典获取
        for name in ['LightGBM', 'CatBoost', 'PrototypicalNet']:
            if name in advanced_models and advanced_models[name] is not None:
                all_models_for_plot.append(advanced_models[name])
                all_model_names_for_plot.append(name)
                print(f'      ✅ {name}: 从advanced_models添加')
            elif name == 'PrototypicalNet':
                # 如果PrototypicalNet不在字典中，尝试从locals直接获取
                print(f'      🔍 PrototypicalNet不在advanced_models中，尝试从locals获取...')
                if 'proto_model' in locals() and proto_model is not None:
                    all_models_for_plot.append(proto_model)
                    all_model_names_for_plot.append(name)
                    print(f'      ✅ PrototypicalNet: 从proto_model变量获取')
                else:
                    print(f'      ⚠️ PrototypicalNet: proto_model变量也不存在')
            else:
                print(f'      ⚠️ {name}: 不在advanced_models中或为None')
    else:
        print(f'   ⚠️ advanced_models不存在或为空，尝试从locals直接获取...')
        # 如果advanced_models不存在，尝试从locals直接获取
        if 'lgb_model' in locals() and lgb_model is not None:
            all_models_for_plot.append(lgb_model)
            all_model_names_for_plot.append('LightGBM')
            print(f'      ✅ LightGBM: 从lgb_model变量获取')
        if 'cat_model' in locals() and cat_model is not None:
            all_models_for_plot.append(cat_model)
            all_model_names_for_plot.append('CatBoost')
            print(f'      ✅ CatBoost: 从cat_model变量获取')
        if 'proto_model' in locals() and proto_model is not None:
            all_models_for_plot.append(proto_model)
            all_model_names_for_plot.append('PrototypicalNet')
            print(f'      ✅ PrototypicalNet: 从proto_model变量获取')
    
    # 确保至少有6个模型（如果缺少，尝试补充）
    expected_models = ['SVM', 'RandomForest', 'DecisionTree', 'LightGBM', 'CatBoost', 'PrototypicalNet']
    missing_models = set(expected_models) - set(all_model_names_for_plot)
    if missing_models:
        print(f'   ⚠️ 警告: 缺少以下模型用于绘图: {missing_models}')
        print(f'   💡 这些模型可能训练失败或未定义')
    
    print(f'   📊 所有模型（用于绘图）: {len(all_models_for_plot)}个 - {all_model_names_for_plot}')
    
    # 然后选择用于Stacking的模型（过滤弱模型）
    all_models = []
    all_model_names = []
    
    # 计算性能阈值：平均性能 - 0.05（移除明显弱于平均的模型）
    if model_performance:
        avg_perf = np.mean(list(model_performance.values()))
        threshold = max(0.5, avg_perf - 0.05)  # 至少保留50%准确率
        print(f'  性能阈值: {threshold:.4f} (平均: {avg_perf:.4f})')
        print(f'  💡 注意: 过滤的模型仍会出现在图表中，但不用于Stacking训练')
        
        # 添加传统模型（过滤弱模型）
        for name, model in [('SVM', svm_model), ('RandomForest', rf_model), ('DecisionTree', dt_model)]:
            if model_performance.get(name, 0) >= threshold:
                all_models.append(model)
                all_model_names.append(name)
                print(f'  ✅ 保留 {name} (性能: {model_performance[name]:.4f}) - 用于Stacking')
            else:
                print(f'  ⚠️ {name} (性能: {model_performance.get(name, 0):.4f} < 阈值) - 仅用于绘图')
        
        # 添加先进模型
        if 'advanced_models' in locals() and advanced_models:
            for name in ['LightGBM', 'CatBoost', 'PrototypicalNet']:
                if name in advanced_models:
                    if model_performance.get(name, 0) >= threshold:
                        all_models.append(advanced_models[name])
                        all_model_names.append(name)
                        print(f'  ✅ 保留 {name} (性能: {model_performance[name]:.4f}) - 用于Stacking')
                    else:
                        print(f'  ⚠️ {name} (性能: {model_performance.get(name, 0):.4f} < 阈值) - 仅用于绘图')
    else:
        # 如果评估失败，使用默认组合（但不包含DecisionTree）
        print('  ⚠️ 模型评估失败，使用默认组合（排除DecisionTree）')
        all_models = [svm_model, rf_model]
        all_model_names = ['SVM', 'RandomForest']
        if 'advanced_models' in locals() and advanced_models:
            for name in ['LightGBM', 'CatBoost', 'PrototypicalNet']:
                if name in advanced_models:
                    all_models.append(advanced_models[name])
                    all_model_names.append(name)
    
    # 确保all_models_for_plot已定义（如果评估失败，使用所有模型）
    if 'all_models_for_plot' not in locals() or len(all_models_for_plot) == 0:
        all_models_for_plot = []
        all_model_names_for_plot = []
        for name, model in [('SVM', svm_model), ('RandomForest', rf_model), ('DecisionTree', dt_model)]:
            all_models_for_plot.append(model)
            all_model_names_for_plot.append(name)
        if 'advanced_models' in locals() and advanced_models:
            for name in ['LightGBM', 'CatBoost', 'PrototypicalNet']:
                if name in advanced_models:
                    all_models_for_plot.append(advanced_models[name])
                    all_model_names_for_plot.append(name)
    
    print(f'\n📊 Stacking使用的模型 ({len(all_models)}个): {", ".join(all_model_names)}')
    print(f'📊 绘图将显示所有模型 ({len(all_models_for_plot)}个): {", ".join(all_model_names_for_plot)}')
    if len(all_models) < 3:
        print('  ⚠️ 警告: 模型数量较少，集成效果可能有限')
    elif len(all_models) > 6:
        print('  ⚠️ 警告: 模型数量较多，可能影响集成效果')
    
    # ============ 移除复杂的模型选择逻辑（已简化） ============
    # 原来的复杂逻辑已被移除，直接使用上面定义的6个基础模型
    
    # 以下是旧代码（已禁用，保留以备参考）
    if False:  # 设为False禁用旧逻辑
        print('\n⚙️ 使用智能模型选择策略')
        print('=' * 60)
        
        # ============ 第1步：评估三大强模型 ============
        print('\n第1步: 评估三大强树集成模型...')
        strong_ensemble_models = {
            'RandomForest': rf_model if 'rf_model' in locals() else None,
            'LightGBM': advanced_models.get('LightGBM') if 'advanced_models' in locals() else None,
            'CatBoost': advanced_models.get('CatBoost') if 'advanced_models' in locals() else None
        }
        
        # 评估性能
        from sklearn.model_selection import train_test_split
        X_mini_test, _, y_mini_test, _ = train_test_split(
            X_test_features, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
        
        performance_ranking = []
        for name, model in strong_ensemble_models.items():
            if model is not None:
                try:
                    if hasattr(model, 'score'):
                        acc = model.score(X_mini_test, y_mini_test)
                    else:
                        pred = model.predict(X_mini_test)
                        acc = (pred == y_mini_test).mean()
                    performance_ranking.append((name, model, acc))
                    print(f'  {name:15s}: {acc:.4f}')
                except Exception as e:
                    print(f'  {name:15s}: 评估失败 ({e})')
        
        # 按性能排序
        performance_ranking.sort(key=lambda x: x[2], reverse=True)
        
        if not performance_ranking:
            print('  ⚠️ 没有可用的强树集成模型，使用默认策略')
            all_models, all_model_names = select_ensemble_models(
                base_models_dict, advanced_models_dict, strategy='diverse'
            )
        else:
            # ============ 第2步：选择模型组合策略 ============
            print('\n第2步: 选择集成策略...')
            
            # 三种策略供选择
            SELECTION_MODE = 'best_one'  # 可选: 'best_one', 'top_two', 'all_three'
            
            print(f'\n当前策略: {SELECTION_MODE}')
            
            if SELECTION_MODE == 'best_one':
                # 策略A: 只用最强的1个 + 非树模型
                print('  策略A: 1个最强树集成 + 非树模型（避免重复集成）')
                
                best_name, best_model, best_acc = performance_ranking[0]
                print(f'  └─ 选中最强树集成: {best_name} ({best_acc:.4f})')
                
                all_models = []
                all_model_names = []
                
                # 添加非树模型
                all_models.append(svm_model)
                all_model_names.append('SVM')
                print(f'  └─ 添加非树模型: SVM (提供线性边界)')
                
                # 添加最强树集成
                all_models.append(best_model)
                all_model_names.append(best_name)
                
                # 添加PrototypicalNet
                if 'advanced_models' in locals() and 'PrototypicalNet' in advanced_models:
                    all_models.append(advanced_models['PrototypicalNet'])
                    all_model_names.append('PrototypicalNet')
                    print(f'  └─ 添加非树模型: PrototypicalNet (提供度量学习)')
                
            elif SELECTION_MODE == 'top_two':
                # 策略B: 用前2强 + 非树模型
                print('  策略B: 2个最强树集成 + 非树模型（平衡性能与多样性）')
                
                all_models = [svm_model]
                all_model_names = ['SVM']
                print(f'  └─ 添加非树模型: SVM')
                
                for i in range(min(2, len(performance_ranking))):
                    name, model, acc = performance_ranking[i]
                    all_models.append(model)
                    all_model_names.append(name)
                    print(f'  └─ 添加树集成: {name} ({acc:.4f})')
                
                if 'advanced_models' in locals() and 'PrototypicalNet' in advanced_models:
                    all_models.append(advanced_models['PrototypicalNet'])
                    all_model_names.append('PrototypicalNet')
                    print(f'  └─ 添加非树模型: PrototypicalNet')
                
            else:  # 'all_three'
                # 策略C: 用所有3个强树集成
                print('  策略C: 所有3个强树集成（最大化性能）')
                
                all_models = []
                all_model_names = []
                
                for name, model, acc in performance_ranking:
                    all_models.append(model)
                    all_model_names.append(name)
                    print(f'  └─ 添加树集成: {name} ({acc:.4f})')
            
            # ============ 第3步：显示最终组合 ============
            print(f'\n🎯 最终选择的模型组合 ({len(all_models)}个):')
            for i, name in enumerate(all_model_names, 1):
                model_type = '树集成' if name in ['RandomForest', 'LightGBM', 'CatBoost'] else '非树模型'
                print(f'  {i}. {name:15s} ({model_type})')
            
            # ============ 第4步：策略说明 ============
            print('\n💡 策略说明:')
            if SELECTION_MODE == 'best_one':
                print(f'  ✅ 只保留1个最强树集成({performance_ranking[0][0]})，避免多个树模型预测相似')
                print(f'  ✅ 添加非树模型(SVM, PrototypicalNet)，增加模型多样性')
                print(f'  ✅ 这样避免"集成的集成"问题，更可能超过单模型')
                print(f'\n  📊 预期: AWEL可能超过{performance_ranking[0][0]} +0.5%~1%')
            elif SELECTION_MODE == 'top_two':
                print(f'  ✅ 选择前2强树集成，保证性能')
                print(f'  ✅ 添加非树模型，保持多样性')
                print(f'  ⚠️  两个树模型可能预测相似，提升有限')
                print(f'\n  📊 预期: AWEL可能与{performance_ranking[0][0]}持平或略高')
            else:
                print(f'  ⚠️  使用所有3个强树集成')
                print(f'  ⚠️  它们可能预测高度相似，集成效果可能有限')
                print(f'  ⚠️  可能无法超过单独的{performance_ranking[0][0]}')
                print(f'\n  📊 预期: AWEL可能略低于或等于{performance_ranking[0][0]}')
            
            print('\n💡 如果想尝试其他策略:')
            print(f"  - 修改第{2380+103}行: SELECTION_MODE = 'best_one'  (推荐)")
            print(f"  - 或: SELECTION_MODE = 'top_two'")
            print(f"  - 或: SELECTION_MODE = 'all_three'")
    
    # ============ 已移除else分支（旧的自动选择逻辑） ============
    # all_models 和 all_model_names 已在前面定义完成
    # y_val_final 和 y_test_final 已在智能模型选择之前定义
    
    # ============ 数据集划分信息 ============
    print(f'\n📊 数据集划分:')
    print(f'  训练集: {X_train_features.shape[0]} 样本')
    print(f'  验证集: {X_val_features.shape[0]} 样本')
    print(f'  测试集(最终评估): {X_test_final_features.shape[0]} 样本')
    
    # ============ 训练Stacking集成方法（使用所有可用模型） ============
    print('\n' + '=' * 60)
    print(f'🔬 训练Stacking集成学习方法（包含{len(all_models)}个基学习器）')
    print('=' * 60)
    
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # 使用模型包装器使所有模型兼容sklearn接口
    print('\n📦 包装所有模型使其兼容Stacking接口...')
    wrapped_estimators = []
    for name, model in zip(all_model_names, all_models):
        wrapped_model = ModelWrapper(model, name)
        wrapped_estimators.append((name, wrapped_model))
        print(f'  ✅ {name}: 已包装')
    
    if len(wrapped_estimators) >= 2:
        print(f'\n🎯 Stacking使用{len(wrapped_estimators)}个基学习器: {[name for name, _ in wrapped_estimators]}')
        
        # 优化Stacking的final_estimator参数
        print('\n🔍 优化Stacking的元学习器参数...')
        
        # 首先使用简单配置快速生成元特征
        print('   步骤1: 使用简单配置生成元特征（用于参数优化）...')
        temp_stacking = StackingClassifier(
            estimators=wrapped_estimators,
            final_estimator=LogisticRegression(max_iter=500, random_state=42),
            cv=3,  # 先用3折快速生成
            n_jobs=-1
        )
        temp_stacking.fit(X_train_features, y_train)
        
        # 获取元特征（基模型的预测概率）
        meta_features = temp_stacking.transform(X_train_features)
        print(f'   元特征维度: {meta_features.shape}')
        
        # 选择简单的元学习器 - 使用基础分类模型
        print('   步骤2: 选择基础元学习器...')
        
        # 简单比较LogisticRegression和RandomForest，选择更好的
        print('   🔍 测试LogisticRegression...')
        lr_simple = LogisticRegression(C=10.0, max_iter=2000, random_state=42)
        lr_score = cross_val_score(lr_simple, meta_features, y_train, cv=5, scoring='accuracy').mean()
        print(f'      ✅ LogisticRegression交叉验证准确率: {lr_score:.4f}')
        
        print('   🔍 测试RandomForest...')
        rf_simple = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_score = cross_val_score(rf_simple, meta_features, y_train, cv=5, scoring='accuracy').mean()
        print(f'      ✅ RandomForest交叉验证准确率: {rf_score:.4f}')
        
        # 选择更好的元学习器
        if rf_score > lr_score:
            print(f'   🎯 选择RandomForest作为元学习器（准确率更高: {rf_score:.4f} vs {lr_score:.4f}）')
            best_final_estimator = rf_simple
            best_meta_name = 'RandomForest'
        else:
            print(f'   🎯 选择LogisticRegression作为元学习器（准确率更高: {lr_score:.4f} vs {rf_score:.4f}）')
            best_final_estimator = lr_simple
            best_meta_name = 'LogisticRegression'
        
        # 优化策略：优先选择表现最好的基学习器
        print('\n📊 分析基学习器性能，优先选择最佳模型组合...')
        
        # 评估每个基学习器在验证集上的性能
        base_learner_performance = {}
        for name, model in zip(all_model_names, all_models):
            try:
                if hasattr(model, 'score'):
                    acc = model.score(X_val_features, y_val_final)
                else:
                    pred = model.predict(X_val_features)
                    acc = (pred == y_val_final).mean()
                base_learner_performance[name] = acc
            except:
                base_learner_performance[name] = 0.0
        
        # 按性能排序
        sorted_learners = sorted(base_learner_performance.items(), key=lambda x: x[1], reverse=True)
        print(f'   基学习器性能排序: {[(name, f"{acc:.4f}") for name, acc in sorted_learners]}')
        
        # 选择表现最好的前N个基学习器（如果模型太多，可能导致元学习器难以学习）
        # 策略：选择Top 4-5个模型，确保包含LightGBM和CatBoost
        top_models = [name for name, _ in sorted_learners[:5]]  # 选择前5个
        print(f'   🎯 选择Top {len(top_models)}个基学习器: {top_models}')
        
        # 重新包装选中的基学习器
        selected_estimators = []
        for name, model in zip(all_model_names, all_models):
            if name in top_models:
                wrapped_model = ModelWrapper(model, name)
                selected_estimators.append((name, wrapped_model))
        
        # Stacking (堆叠集成) - 使用优化后的final_estimator和精选的基学习器
        print(f'\n🚀 训练优化后的Stacking（使用{len(selected_estimators)}个精选基学习器 + 强元学习器）...')
        try:
            stacking = StackingClassifier(
                estimators=selected_estimators,  # 使用精选的基学习器
                final_estimator=best_final_estimator,
                cv=7,  # 增加交叉验证折数以获得更稳定的元特征
                n_jobs=-1,
                verbose=1  # 显示训练进度
            )
            stacking.fit(X_train_features, y_train)
            stacking_acc = stacking.score(X_val_features, y_val_final)
            print(f'\n✅ Stacking训练完成！验证准确率: {stacking_acc:.4f}')
            print(f'   使用的基学习器: {", ".join([name for name, _ in selected_estimators])}')
            print(f'   使用的元学习器: {best_meta_name} ({type(best_final_estimator).__name__})')
        except Exception as e:
            print(f'\n⚠️ Stacking训练失败: {e}')
            print('   尝试使用简化的优化配置...')
            try:
                # 回退方案：使用优化的LogisticRegression但减少cv折数
                best_lr_simple = LogisticRegression(
                    C=10.0,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=2000,
                    random_state=42
                )
                stacking = StackingClassifier(
                    estimators=selected_estimators if 'selected_estimators' in locals() else wrapped_estimators,
                    final_estimator=best_lr_simple,
                    cv=5,
                    n_jobs=-1
                )
                stacking.fit(X_train_features, y_train)
                stacking_acc = stacking.score(X_val_features, y_val_final)
                print(f'✅ Stacking回退方案成功！验证准确率: {stacking_acc:.4f}')
                print(f'   使用的基学习器: {", ".join([name for name, _ in wrapped_estimators])}')
            except Exception as e2:
                print(f'\n⚠️ Stacking回退方案也失败: {e2}')
                print('   尝试使用sklearn原生兼容的模型...')
                # 最后回退到只使用sklearn兼容的模型
                sklearn_compatible = ['SVM', 'RandomForest', 'DecisionTree']
                fallback_estimators = [(name, model) for name, model in zip(all_model_names, all_models) 
                                      if name in sklearn_compatible]
                if len(fallback_estimators) >= 2:
                    try:
                        stacking = StackingClassifier(
                            estimators=fallback_estimators,
                            final_estimator=LogisticRegression(C=10.0, max_iter=2000, random_state=42),
                            cv=5,
                            n_jobs=-1
                        )
                        stacking.fit(X_train_features, y_train)
                        stacking_acc = stacking.score(X_val_features, y_val_final)
                        print(f'✅ Stacking最终回退方案成功！验证准确率: {stacking_acc:.4f}')
                        print(f'   使用的基学习器: {", ".join([name for name, _ in fallback_estimators])}')
                    except Exception as e3:
                        print(f'❌ Stacking最终回退方案也失败: {e3}')
                        print('❌ Stacking训练失败，跳过')
                        stacking = None
                        stacking_acc = 0
                else:
                    print('❌ Stacking训练失败，跳过（可用模型不足）')
                    stacking = None
                    stacking_acc = 0
    else:
        print('⚠️ 可用模型少于2个，跳过Stacking')
        stacking = None
        stacking_acc = 0
    
    # ============ 评估Stacking集成方法 ============
    if stacking is not None:
        print('\n' + '=' * 60)
        print('📊 评估Stacking集成方法（测试集）')
        print('=' * 60)
        
        result_stacking = evaluate_model(
            stacking,
            X_train_features, y_train,
            X_test_final_features, y_test_final,
            'Stacking',
            file_info
        )
        results_dict['Stacking'] = result_stacking
    else:
        print('\n⚠️ 跳过Stacking评估（sklearn兼容模型不足）')
    
    # ================= 对比分析：Stacking vs 最佳单模型 ===============
    print('\n' + '=' * 60)
    print('🏆 性能对比分析')
    print('=' * 60)
    
    # 找出最佳单模型
    best_single_model = None
    best_single_acc = 0
    best_single_name = ''
    
    for name in all_model_names:
        if name in results_dict:
            acc = results_dict[name]['test']['accuracy']
            if acc > best_single_acc:
                best_single_acc = acc
                best_single_name = name
    
    if stacking is not None and 'Stacking' in results_dict:
        stacking_acc = results_dict['Stacking']['test']['accuracy']
        improvement = (stacking_acc - best_single_acc) * 100
        
        print(f'\n📊 性能提升分析:')
        print(f'  最佳单模型: {best_single_name} = {best_single_acc:.4f}')
        print(f'  Stacking集成: {stacking_acc:.4f}')
        print(f'  提升幅度: {"+" if improvement >= 0 else ""}{improvement:.2f}%')
        
        if improvement > 0:
            print(f'\n🎉 集成学习成功！Stacking比最佳单模型提升了{improvement:.2f}%')
        else:
            print(f'\n💡 提示: Stacking未超过最佳单模型，可能的原因：')
            print(f'   - 数据集较小，集成学习收益有限')
            print(f'   - 最佳单模型（CatBoost/RandomForest）本身已是集成模型')
            print(f'   - 基学习器之间互补性不足')
    else:
        print('\n⚠️ Stacking未成功训练，无法进行性能对比')
    
    # 准备所有模型用于绘图（所有6个基础模型 + Stacking）
    print('\n' + '=' * 60)
    print('📊 准备绘制模型列表')
    print('=' * 60)
    
    # 使用所有模型进行绘图（包括被过滤的模型），确保始终显示6+1=7个模型
    models = all_models_for_plot.copy()
    model_names_plot = all_model_names_for_plot.copy()
    
    print(f'\n基础模型 ({len(all_models_for_plot)}个，包含所有模型用于绘图):')
    for i, name in enumerate(all_model_names_for_plot, 1):
        # 标记哪些模型用于Stacking
        stacking_marker = ' (用于Stacking)' if name in all_model_names else ' (仅绘图)'
        print(f'  {i}. {name}{stacking_marker}')
    
    # 添加Stacking模型（如果训练成功）
    print(f'\n集成模型:')
    if stacking is not None:
        models.append(stacking)
        model_names_plot.append('Stacking')
        print(f'  ✅ Stacking集成模型已添加（使用{len(wrapped_estimators) if "wrapped_estimators" in locals() else len(all_models)}个基学习器）')
    else:
        print(f'  ⚠️ Stacking未训练成功（stacking=None），不会包含在图表中')
        print(f'     可能原因:')
        print(f'     1. Stacking训练过程中出错')
        print(f'     2. 模型数量不足（需要至少2个模型）')
        print(f'     3. 查看上面的Stacking训练输出了解详情')
    
    print(f'\n📊 最终将绘制 {len(models)} 个模型: {model_names_plot}')
    print(f'   - 基础模型: {len(all_models_for_plot)}个 ({", ".join(all_model_names_for_plot)})')
    if stacking is not None:
        print(f'   - 集成模型: 1个 (Stacking)')
        print(f'   ✅ 总计: {len(models)}个模型将被绘制 (期望: 6+1=7个)')
    else:
        print(f'   ⚠️ 集成模型: 0个 (Stacking训练失败或未训练)')
        print(f'   ⚠️ 注意: 只有{len(all_models_for_plot)}个单模型，缺少Stacking集成模型')
        print(f'   💡 提示: 如果期望看到7个模型，请检查Stacking训练是否成功')
    
    # ================= 汇总所有模型性能 ===============
    print('\n' + '=' * 60)
    print('📊 汇总所有模型性能（包括Stacking集成模型）')
    print('=' * 60)
    
    summary_df = summarize_results(results_dict)
    summary_df.to_csv("model_summary.csv", index=False)
    print('\n✅ 性能汇总已保存至: model_summary.csv')
    
    # ================= 绘制所有模型的可视化图表 ===============
    print('\n' + '=' * 60)
    print('📊 绘制所有模型的混淆矩阵和ROC曲线')
    print('=' * 60)
    
    # 绘制混淆矩阵和ROC（使用最终测试集）
    # 在绘图前验证所有模型
    print('\n🔍 验证所有模型状态...')
    verified_models = []
    verified_names = []
    for model, name in zip(models, model_names_plot):
        try:
            # 检查模型是否有predict方法
            has_predict = hasattr(model, "predict")
            has_forward = hasattr(model, "forward")
            
            if has_predict or has_forward:
                # 尝试进行一次预测测试（使用小样本）
                test_sample = X_test_final_features[:1] if len(X_test_final_features) > 0 else X_train_features[:1]
                if has_predict:
                    _ = model.predict(test_sample)
                elif has_forward:
                    import torch
                    model.eval()
                    with torch.no_grad():
                        _ = model.forward(torch.FloatTensor(test_sample))
                
                verified_models.append(model)
                verified_names.append(name)
                print(f'  ✅ {name}: 验证通过')
            else:
                print(f'  ⚠️ {name}: 没有predict或forward方法，跳过')
        except Exception as e:
            print(f'  ❌ {name}: 验证失败 - {e}，跳过')
    
    print(f'\n📊 最终验证通过 {len(verified_models)} 个模型: {verified_names}')
    if len(verified_models) != len(models):
        print(f'  ⚠️ 警告: {len(models)}个模型中有{len(models) - len(verified_models)}个未通过验证')
        print(f'  💡 建议: 检查被跳过的模型的训练状态')
    
    # ====== 专门对 CatBoost 做一次“手工”混淆矩阵与准确率检查 ======
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix
        for vm, vname in zip(verified_models, verified_names):
            # 名称中包含 "CatBoost" 就认为是 CatBoost 模型
            if 'CatBoost' in vname:
                print('\n🐱 [Debug] 使用最终测试集，单独检查 CatBoost 的预测情况...')
                y_pred_cat = vm.predict(X_test_final_features)
                acc_cat = accuracy_score(y_test_final, y_pred_cat)
                cm_cat = confusion_matrix(y_test_final, y_pred_cat)
                print('CatBoost 在最终测试集上的 Accuracy:', acc_cat)
                print('CatBoost 在最终测试集上的混淆矩阵：')
                print(cm_cat)
                break
    except Exception as e:
        print(f'⚠️ CatBoost 手工检查时出错: {e}')
    
    # 使用验证通过的模型进行绘图
    if len(verified_models) > 0:
        plot_confusion_matrix_and_roc(verified_models, verified_names, X_train_features, y_train, 'Train Set')
        plot_confusion_matrix_and_roc(verified_models, verified_names, X_test_final_features, y_test_final, 'Test Set')
    else:
        print('\n❌ 没有可用的模型进行绘图！')

    # =============== 可解释性分析阶段 ===============
    # 将每条扩充后的轨迹分别划分为六个阶段
    all_phases = [split_into_phases(trajectory) for trajectory in X_test]
    num = 6
    for phases in all_phases:
        assert len(phases) == num, f"Expected 6 phases but got {len(phases)}"

    # 使用分割好的阶段数据绘制并保存3D图像
    plot_phases_3d(all_phases, y, file_info, base_save_path='3D_Trajectories_new_Phases', num_phases=num)

    phases_by_stage = [[] for _ in range(num)]  # 存放每个阶段的所有特征

    # 将每个阶段的特征组合起来，便于后续 SHAP 处理
    for phases in all_phases:
        for i, phase in enumerate(phases):
            phases_by_stage[i].extend(phase)

    # 时间维度 T = 200
    T = X_test.shape[1]
    # 把 [0,1,…,199] 平均拆成 6 段
    phases_time = np.array_split(np.arange(T), 6)
    # 对每个阶段，把它所有时间点 t 转成扁平后的 3 个维度索引 [3*t,3*t+1,3*t+2]
    phases_feature_indices = [
        [i for t in phase for i in (3 * t, 3 * t + 1, 3 * t + 2)]
        for phase in phases_time
    ]

    # 确保每个阶段的数据形状正确
    for i, phase_data in enumerate(phases_by_stage):
        print(f"Phase {i + 1} has {len(phase_data)} samples")

    # ============ SHAP分析：分为基础模型和集成模型 ============
    print('\n' + '=' * 60)
    print('📊 SHAP可解释性分析')
    print('=' * 60)
    
    # 第1步: 基础模型的SHAP分析（使用原始轨迹数据）
    print(f'\n1️⃣ 基础模型SHAP分析（{len(all_models)}个模型）...')
    base_models_for_shap = all_models.copy()
    base_names_for_shap = all_model_names.copy()
    
    shap_values_base = plot_phase_shap_importance(
        base_models_for_shap,
        base_names_for_shap,
        X_train, y_train, X_test, 
        phases_feature_indices
    )
    
    # 第2步: 集成模型的SHAP分析（使用LSTM特征）
    print('\n2️⃣ 集成模型SHAP分析（Stacking）...')
    ensemble_models = []
    ensemble_names = []
    
    if stacking is not None:
        ensemble_models.append(stacking)
        ensemble_names.append('Stacking')
    
    # 为集成模型创建简化的SHAP分析
    # 注意：集成模型使用的是32维LSTM特征，不是原始轨迹
    shap_values_ensemble = plot_ensemble_shap_importance(
        ensemble_models,
        ensemble_names,
        X_train_features, y_train,
        X_test_final_features, y_test_final
    )
    
    # 第3步: 绘制基础模型的阶段聚合特征重要性图
    print('\n3️⃣ 绘制阶段聚合特征重要性...')
    phases_name = [f'Phase {i + 1}' for i in range(6)]
    plot_phase_aggregated_feature_importance(
        shap_values_base,
        phase_names=phases_name
    )

    # 第4步: 本地SHAP的分类别阶段分析（健康、ADDwR、ADDWoR）
    print('\n4️⃣ 本地SHAP分类别阶段分析...')
    local_class_names = {
        0: 'Healthy',
        1: 'ADDWoR',
        2: 'ADDwR'
    }
    local_class_order = [0, 2, 1]
    classwise_shap_results = plot_classwise_phase_shap_analysis(
        shap_values_base,
        y_test_for_shap,
        class_names=local_class_names,
        class_order=local_class_order,
        save_dir='results/shap_local'
    )

    # 第5步: 本地SHAP的空间方向分析（x/y/z）
    print('\n5️⃣ 本地SHAP空间维度分析（x/y/z）...')
    spatial_shap_results = plot_spatial_dimension_shap_analysis(
        shap_values_base,
        y_test=y_test_for_shap,
        class_names=local_class_names,
        class_order=local_class_order,
        save_dir='results/shap_local'
    )

    # 综合阶段可解释性分析
    print('\n' + '=' * 60)
    print('🚀 开始综合阶段可解释性分析')
    print('=' * 60)
    
    # 准备轨迹数据（使用原始轨迹数据）
    trajectories = X  # 使用已加载的轨迹数据
    labels = y  # 使用已加载的标签数据
    
    # 调用综合阶段分析
    phase_analysis_results = comprehensive_phase_analysis(
        trajectories=trajectories,
        labels=labels,
        models=models,
        model_names=model_names_plot,
        num_phases=6,
        save_path='results/plots'
    )


