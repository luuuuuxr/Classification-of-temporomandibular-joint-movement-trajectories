# -*- coding: utf-8 -*-
"""
贝叶斯调参模块：
  bayesian_optimization_cv / train_models_with_best_params
  _make_serializable / save_best_params_to_file / load_best_params_from_file
"""

import copy
import json
import os

import numpy as np
from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.optimization.hyperopt_config import (
    MODEL_CONFIGS, BEST_PARAM_CACHE_PATH, DEFAULT_FIXED_MODEL_PARAMS,
    LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE
)
from src.models.classifiers import PrototypicalNetwork

try:
    import lightgbm as lgb
except ImportError:
    pass

try:
    import catboost as cb
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────
#  参数序列化 / 持久化
# ──────────────────────────────────────────────────────────────

def _make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_best_params_to_file(params, path=BEST_PARAM_CACHE_PATH):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_make_serializable(params), f, indent=2)


def load_best_params_from_file(path=BEST_PARAM_CACHE_PATH):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f'⚠️ 无法解析已保存的参数文件：{path}')
        return None


# ──────────────────────────────────────────────────────────────
#  贝叶斯调参
# ──────────────────────────────────────────────────────────────

def bayesian_optimization_cv(model_type, X, y, class_weight=None,
                              max_evals=120, n_splits=10, fast_mode=False):
    config = MODEL_CONFIGS[model_type]
    space, map_func = config['space'], config['map']
    if fast_mode:
        n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    trial_history = []

    def objective(params):
        try:
            mapped = map_func(params)

            if model_type == 'PrototypicalNet':
                from sklearn.metrics import accuracy_score
                scores = []
                input_dim = X.shape[1]
                for train_idx, val_idx in kf.split(X, y):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    proto = PrototypicalNetwork(
                        input_dim=input_dim,
                        hidden_dim=int(mapped['hidden_dim']),
                        output_dim=int(mapped['output_dim'])
                    )
                    proto.fit(X_train_fold, y_train_fold,
                              epochs=int(mapped['epochs']),
                              batch_size=int(mapped['batch_size']),
                              lr=float(mapped['lr']))
                    y_pred = proto.predict(X_val_fold)
                    scores.append(accuracy_score(y_val_fold, y_pred))
                score = float(np.mean(scores))
                trial_history.append({'params': mapped, 'score': score})
                return {'loss': -score, 'status': STATUS_OK}

            if model_type == 'SVM':
                model = SVC(**mapped, probability=True, random_state=42,
                            class_weight=class_weight)
            elif model_type == 'RandomForest':
                model = RandomForestClassifier(**mapped, random_state=42,
                                              class_weight=class_weight)
            elif model_type == 'DecisionTree':
                model = DecisionTreeClassifier(**mapped, random_state=42,
                                               class_weight=class_weight)
            elif model_type == 'LightGBM':
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM 不可用")
                model = lgb.LGBMClassifier(
                    objective='multiclass', random_state=42,
                    class_weight=class_weight, **mapped)
            elif model_type == 'CatBoost':
                if not CATBOOST_AVAILABLE:
                    raise ImportError("CatBoost 不可用")
                class_weights = None
                if class_weight:
                    sorted_labels = sorted(np.unique(y))
                    class_weights = [class_weight.get(label, 1.0)
                                     for label in sorted_labels]
                catboost_params = mapped.copy()
                if 'border_count' in catboost_params:
                    catboost_params['border_count'] = max(
                        1, int(catboost_params['border_count']))
                catboost_params['iterations'] = max(
                    1, int(catboost_params.get('iterations', 100)))
                model = cb.CatBoostClassifier(
                    loss_function='MultiClass', eval_metric='Accuracy',
                    verbose=0, allow_writing_files=False,
                    class_weights=class_weights, **catboost_params)
            else:
                raise ValueError("Unknown model type")

            score = cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
            trial_history.append({'params': mapped, 'score': score})
            return {'loss': -score, 'status': STATUS_OK}

        except Exception as e:
            print(f"[{model_type} ERROR] {e}")
            trial_history.append(
                {'params': mapped if 'mapped' in locals() else {},
                 'score': None, 'error': str(e)})
            return {'loss': 1.0, 'status': STATUS_OK}

    trials = Trials()
    best_index_params = fmin(fn=objective, space=space, algo=tpe.suggest,
                             max_evals=max_evals, trials=trials)
    best_mapped_params = map_func(best_index_params)
    return best_mapped_params, trial_history


# ──────────────────────────────────────────────────────────────
#  模型训练入口（调参 or 固定参数）
# ──────────────────────────────────────────────────────────────

def train_models_with_best_params(X_train_features, y_train, weight,
                                  mode='search',
                                  param_cache_path=BEST_PARAM_CACHE_PATH,
                                  predefined_params=None):
    from src.evaluation.plots import plot_param_search_diagnostics, plot_learning_curve

    trained_models = {}
    advanced_models = {}
    mode = (mode or 'search').lower()
    if mode not in ['search', 'fixed']:
        print(f'⚠️ 未知模式 {mode}，自动切换为 search。')
        mode = 'search'

    model_sequence = ['SVM', 'RandomForest', 'DecisionTree']
    if LIGHTGBM_AVAILABLE:
        model_sequence.append('LightGBM')
    if CATBOOST_AVAILABLE:
        model_sequence.append('CatBoost')

    FAST_MODE = False

    cached_params = {}
    if mode == 'fixed':
        if predefined_params:
            cached_params.update(copy.deepcopy(predefined_params))
        loaded = load_best_params_from_file(param_cache_path)
        if loaded:
            cached_params.update(loaded)
        if not cached_params:
            print('⚠️ 未找到任何固定参数，将自动退回搜索模式。')
            mode = 'search'

    best_param_log = {}

    if mode == 'search':
        print('\n🚀 开始执行全模型贝叶斯调参（SVM / RF / DT / LightGBM / CatBoost）...')
    else:
        print('\n🚀 使用预设/缓存参数直接训练模型（跳过贝叶斯调参）...')

    for model_type in model_sequence:
        if mode == 'search':
            print(f'\n🔍 正在优化模型: {model_type}')
            if FAST_MODE:
                max_evals = 30 if model_type not in ['LightGBM', 'CatBoost'] else 20
                if model_type == 'SVM':
                    max_evals = 25
            else:
                max_evals = 90
                if model_type in ['LightGBM', 'CatBoost']:
                    max_evals = 60
                elif model_type == 'SVM':
                    max_evals = 80

            best_params, trial_history = bayesian_optimization_cv(
                model_type, X_train_features, y_train,
                class_weight=weight, max_evals=max_evals, fast_mode=FAST_MODE)
            print(f'✅ 最佳参数 ({model_type}): {best_params}')
            plot_param_search_diagnostics(model_type, trial_history)
            best_param_log[model_type] = copy.deepcopy(best_params)
        else:
            best_params = copy.deepcopy(cached_params.get(model_type, {}))
            if not best_params:
                print(f'⚠️ 未找到 {model_type} 的固定参数，跳过该模型。')
                continue
            print(f'✅ 使用固定参数 ({model_type}): {best_params}')

        if model_type == 'SVM':
            model = SVC(**best_params, probability=True, random_state=42,
                        class_weight=weight)
        elif model_type == 'RandomForest':
            model = RandomForestClassifier(**best_params, random_state=42,
                                           class_weight=weight)
        elif model_type == 'DecisionTree':
            model = DecisionTreeClassifier(**best_params, random_state=42,
                                           class_weight=weight)
        elif model_type == 'LightGBM':
            model = lgb.LGBMClassifier(objective='multiclass', random_state=42,
                                        class_weight=weight, **best_params)
        elif model_type == 'CatBoost':
            sorted_labels = sorted(np.unique(y_train))
            class_weights = [weight.get(label, 1.0) for label in sorted_labels]
            catboost_params = best_params.copy()
            if 'border_count' in catboost_params:
                catboost_params['border_count'] = max(
                    1, int(catboost_params['border_count']))
            catboost_params['iterations'] = max(
                1, int(catboost_params.get('iterations', 100)))
            model = cb.CatBoostClassifier(
                loss_function='MultiClass', eval_metric='Accuracy',
                verbose=0, allow_writing_files=False,
                class_weights=class_weights, **catboost_params)
        else:
            continue

        model.fit(X_train_features, y_train)

        if mode == 'search':
            try:
                plot_learning_curve(model, X_train_features, y_train, model_type)
            except Exception as plot_err:
                print(f'⚠️ 学习曲线绘制失败（{model_type}）: {plot_err}')

        if model_type in ['SVM', 'RandomForest', 'DecisionTree']:
            trained_models[model_type] = model
        else:
            advanced_models[model_type] = model

    # PrototypicalNet
    try:
        input_dim = X_train_features.shape[1]
        if mode == 'search':
            print('\n🚀 开始执行 PrototypicalNet 的贝叶斯调参...')
            max_evals_proto = 10 if FAST_MODE else 30
            best_params_proto, trial_history_proto = bayesian_optimization_cv(
                'PrototypicalNet', X_train_features, y_train,
                class_weight=None, max_evals=max_evals_proto, fast_mode=FAST_MODE)
            print(f'✅ 最佳参数 (PrototypicalNet): {best_params_proto}')
            plot_param_search_diagnostics('PrototypicalNet', trial_history_proto)
            best_param_log['PrototypicalNet'] = copy.deepcopy(best_params_proto)
        else:
            best_params_proto = copy.deepcopy(
                cached_params.get('PrototypicalNet',
                                  DEFAULT_FIXED_MODEL_PARAMS['PrototypicalNet']))
            print(f'\n✅ 使用固定参数训练 PrototypicalNet: {best_params_proto}')

        proto_model = PrototypicalNetwork(
            input_dim=input_dim,
            hidden_dim=best_params_proto['hidden_dim'],
            output_dim=best_params_proto['output_dim'],
        )
        proto_model.fit(X_train_features, y_train,
                        epochs=best_params_proto['epochs'],
                        batch_size=best_params_proto['batch_size'],
                        lr=best_params_proto['lr'])
        advanced_models['PrototypicalNet'] = proto_model
        print('✅ Prototypical Network 训练完成')
    except Exception as e:
        print(f'❌ Prototypical Network 训练失败: {e}')

    if mode == 'search' and best_param_log:
        try:
            save_best_params_to_file(best_param_log, param_cache_path)
            print(f'💾 已将最佳参数保存至 {param_cache_path}')
        except Exception as e:
            print(f'⚠️ 最佳参数保存失败: {e}')

    print(f'\n🎉 训练完成！共训练了 {len(trained_models) + len(advanced_models)} 个模型')
    print(f'基础模型: {list(trained_models.keys())}')
    print(f'先进模型: {list(advanced_models.keys())}')

    svm_model = trained_models.get('SVM')
    rf_model = trained_models.get('RandomForest')
    dt_model = trained_models.get('DecisionTree')
    return svm_model, rf_model, dt_model, advanced_models
