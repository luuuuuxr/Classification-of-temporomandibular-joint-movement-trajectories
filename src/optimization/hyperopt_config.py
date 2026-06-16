# -*- coding: utf-8 -*-
"""
超参数搜索配置：MODEL_CONFIGS（各模型的搜索空间 + 映射函数）
              + 固定参数集（标准流程 / sklearn默认值）
"""

import os
import numpy as np
from hyperopt import hp

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
#  超参数搜索空间配置
# ──────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'SVM': {
        'space': {
            'C': hp.loguniform('C', np.log(1), np.log(10)),
            'kernel_index': hp.choice('kernel_index', [0, 1]),
            'gamma_index': hp.choice('gamma_index', [0, 1])
        },
        'map': lambda p: {
            'C': p['C'],
            'kernel': ['linear', 'rbf'][p['kernel_index']],
            'gamma': ['scale', 'auto'][p['gamma_index']]
        }
    },
    'RandomForest': {
        'space': {
            'RF_n_estimators_index': hp.choice('RF_n_estimators_index', [0, 1, 2]),
            'RF_max_depth_index': hp.choice('RF_max_depth_index', [0, 1, 2]),
            'RF_min_samples_leaf': hp.uniform('RF_min_samples_leaf', 0.005, 0.05),
            'RF_min_samples_split': hp.uniform('RF_min_samples_split', 0.01, 0.05),
            'RF_max_features_index': hp.choice('RF_max_features_index', [0, 1]),
            'RF_bootstrap_index': hp.choice('RF_bootstrap_index', [0, 1])
        },
        'map': lambda p: {
            'n_estimators': [40, 50, 60][p['RF_n_estimators_index']],
            'max_depth': [30, 40, 50][p['RF_max_depth_index']],
            'min_samples_leaf': p['RF_min_samples_leaf'],
            'min_samples_split': p['RF_min_samples_split'],
            'max_features': ['sqrt', 'log2'][p['RF_max_features_index']],
            'bootstrap': [True, False][p['RF_bootstrap_index']]
        }
    },
    'DecisionTree': {
        'space': {
            'max_depth_index': hp.choice('max_depth_index', [0, 1, 2]),
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.05),
            'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.05),
            'criterion_index': hp.choice('criterion_index', [0, 1]),
            'splitter_index': hp.choice('splitter_index', [0, 1])
        },
        'map': lambda p: {
            'max_depth': [10, 20, 30][p['max_depth_index']],
            'min_samples_leaf': p['min_samples_leaf'],
            'min_samples_split': p['min_samples_split'],
            'criterion': ['gini', 'entropy'][p['criterion_index']],
            'splitter': ['best', 'random'][p['splitter_index']]
        }
    },
}

if LIGHTGBM_AVAILABLE:
    MODEL_CONFIGS['LightGBM'] = {
        'space': {
            'lgb_num_leaves': hp.quniform('lgb_num_leaves', 16, 64, 1),
            'lgb_learning_rate': hp.loguniform('lgb_learning_rate', np.log(0.01), np.log(0.2)),
            'lgb_max_depth_index': hp.choice('lgb_max_depth_index', [0, 1, 2, 3]),
            'lgb_feature_fraction': hp.uniform('lgb_feature_fraction', 0.6, 1.0),
            'lgb_bagging_fraction': hp.uniform('lgb_bagging_fraction', 0.6, 1.0),
            'lgb_bagging_freq': hp.choice('lgb_bagging_freq', [1, 5, 10]),
            'lgb_min_child_samples': hp.quniform('lgb_min_child_samples', 10, 60, 1),
            'lgb_lambda_l1': hp.loguniform('lgb_lambda_l1', np.log(1e-4), np.log(10)),
            'lgb_lambda_l2': hp.loguniform('lgb_lambda_l2', np.log(1e-4), np.log(10)),
            'lgb_num_boost_round': hp.quniform('lgb_num_boost_round', 50, 150, 5)
        },
        'map': lambda p: {
            'num_leaves': int(p['lgb_num_leaves']),
            'learning_rate': p['lgb_learning_rate'],
            'max_depth': [-1, 16, 32, 48][p['lgb_max_depth_index']],
            'feature_fraction': p['lgb_feature_fraction'],
            'bagging_fraction': p['lgb_bagging_fraction'],
            'bagging_freq': p['lgb_bagging_freq'],
            'min_child_samples': int(p['lgb_min_child_samples']),
            'reg_alpha': p['lgb_lambda_l1'],
            'reg_lambda': p['lgb_lambda_l2'],
            'n_estimators': int(p['lgb_num_boost_round'])
        }
    }

if CATBOOST_AVAILABLE:
    MODEL_CONFIGS['CatBoost'] = {
        'space': {
            'cb_iterations': hp.choice('cb_iterations', [80, 140, 200, 260]),
            'cb_depth': hp.choice('cb_depth', [4, 6, 8]),
            'cb_learning_rate': hp.loguniform('cb_learning_rate', np.log(0.01), np.log(0.2)),
            'cb_l2_leaf_reg': hp.loguniform('cb_l2_leaf_reg', np.log(1), np.log(20)),
            'cb_bagging_temp': hp.uniform('cb_bagging_temp', 0.0, 1.0),
            'cb_border_count': hp.choice('cb_border_count', [32, 64, 128, 254])
        },
        'map': lambda p: {
            'iterations': int(p['cb_iterations']),
            'depth': int(p['cb_depth']),
            'learning_rate': float(p['cb_learning_rate']),
            'l2_leaf_reg': float(p['cb_l2_leaf_reg']),
            'bagging_temperature': float(p['cb_bagging_temp']),
            'border_count': int(p['cb_border_count'])
        }
    }

# PrototypicalNet 搜索空间
MODEL_CONFIGS['PrototypicalNet'] = {
    'space': {
        'proto_hidden_dim': hp.choice('proto_hidden_dim', [128, 192, 256]),
        'proto_output_dim': hp.choice('proto_output_dim', [64, 96, 128]),
        'proto_lr': hp.loguniform('proto_lr', np.log(1e-4), np.log(5e-3)),
        'proto_epochs': hp.quniform('proto_epochs', 30, 80, 10),
        'proto_batch_size': hp.choice('proto_batch_size', [16, 32, 64]),
    },
    'map': lambda p: {
        'hidden_dim': int(p['proto_hidden_dim']),
        'output_dim': int(p['proto_output_dim']),
        'lr': float(p['proto_lr']),
        'epochs': int(p['proto_epochs']),
        'batch_size': int(p['proto_batch_size']),
    }
}

BEST_PARAM_CACHE_PATH = os.path.join('results', 'best_hyperparams.json')

# ──────────────────────────────────────────────────────────────
#  固定参数集
# ──────────────────────────────────────────────────────────────

DEFAULT_FIXED_MODEL_PARAMS = {
    'SVM': {'C': 2.0, 'kernel': 'rbf', 'gamma': 'scale'},
    'RandomForest': {
        'n_estimators': 30, 'max_depth': 12,
        'min_samples_leaf': 0.015, 'min_samples_split': 0.03,
        'max_features': 'sqrt', 'bootstrap': True
    },
    'DecisionTree': {
        'max_depth': 10, 'min_samples_leaf': 0.01,
        'min_samples_split': 0.02, 'criterion': 'entropy', 'splitter': 'random'
    },
    'LightGBM': {
        'num_leaves': 63, 'learning_rate': 0.05, 'max_depth': 8,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 3,
        'min_child_samples': 15, 'reg_alpha': 0.1, 'reg_lambda': 0.05,
        'n_estimators': 200
    },
    'CatBoost': {
        'iterations': 300, 'depth': 6, 'learning_rate': 0.05,
        'l2_leaf_reg': 3.0, 'bagging_temperature': 0.5, 'border_count': 128
    },
    'PrototypicalNet': {
        'hidden_dim': 256, 'output_dim': 128,
        'lr': 0.001, 'epochs': 50, 'batch_size': 32
    }
}

# 消融实验 w/o TPE 用的超参：sklearn/库的真实默认值
SKLEARN_DEFAULT_PARAMS = {
    'SVM': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
    'RandomForest': {
        'n_estimators': 100, 'max_depth': None,
        'min_samples_leaf': 1, 'min_samples_split': 2,
        'max_features': 'sqrt', 'bootstrap': True
    },
    'DecisionTree': {
        'max_depth': None, 'min_samples_leaf': 1,
        'min_samples_split': 2, 'criterion': 'gini', 'splitter': 'best'
    },
    'LightGBM': {
        'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 100,
        'max_depth': -1, 'feature_fraction': 1.0, 'bagging_fraction': 1.0,
        'bagging_freq': 0, 'min_child_samples': 20,
        'reg_alpha': 0.0, 'reg_lambda': 0.0
    },
    'CatBoost': {
        'iterations': 500, 'depth': 6, 'learning_rate': 0.03,
        'l2_leaf_reg': 3.0, 'bagging_temperature': 1.0, 'border_count': 128
    },
    'PrototypicalNet': {
        'hidden_dim': 256, 'output_dim': 128,
        'lr': 0.001, 'epochs': 50, 'batch_size': 32
    }
}
