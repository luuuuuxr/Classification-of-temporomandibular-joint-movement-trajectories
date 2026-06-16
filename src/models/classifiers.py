# -*- coding: utf-8 -*-
"""
分类器模块：LightGBM / CatBoost / PrototypicalNetwork / ModelWrapper

这些分类器作为 Stacking 集成的基学习器使用。
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

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

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
#  LightGBM / CatBoost
# ──────────────────────────────────────────────────────────────

class LightGBMClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.classes_ = None
        self.class_to_index_ = None

    def fit(self, X, y):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError('LightGBM not available')
        X_scaled = self.scaler.fit_transform(X)
        self.classes_ = np.unique(y)
        self.class_to_index_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_mapped = np.array([self.class_to_index_[label] for label in y], dtype=int)
        train_data = lgb.Dataset(X_scaled, label=y_mapped)
        params = {
            'objective': 'multiclass',
            'num_class': len(self.classes_),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 10,
            'verbose': -1
        }
        self.model = lgb.train(params, train_data, 80)
        return self

    def predict(self, X):
        pred_proba = self.predict_proba(X)
        pred_idx = np.argmax(pred_proba, axis=1)
        if self.classes_ is None:
            return pred_idx
        return self.classes_[pred_idx]

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        pred_proba = self.model.predict(X_scaled, num_iteration=self.model.best_iteration)
        if pred_proba.ndim == 1:
            pred_proba = np.vstack([1.0 - pred_proba, pred_proba]).T
        return pred_proba


class CatBoostClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        if not CATBOOST_AVAILABLE:
            raise ImportError('CatBoost not available')
        X_scaled = self.scaler.fit_transform(X)
        self.model = cb.CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function='MultiClass',
            random_seed=42
        )
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions.flatten() if predictions.ndim > 1 else predictions

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


# ──────────────────────────────────────────────────────────────
#  Prototypical Network（PyTorch 实现）
# ──────────────────────────────────────────────────────────────

class PrototypicalNetwork:
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        self.prototypes = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )

    def _compute_prototypes(self, embeddings, labels):
        unique_labels = torch.unique(labels)
        prototypes = []
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                prototype = embeddings[mask].mean(dim=0)
                prototypes.append(prototype)
        return torch.stack(prototypes) if prototypes else torch.empty(0, embeddings.size(1))

    def _prototypical_loss(self, embeddings, labels, prototypes):
        if len(prototypes) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        unique_labels = torch.unique(labels)
        label_to_proto_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
        mapped_labels = torch.tensor(
            [label_to_proto_idx[label.item()] for label in labels],
            device=labels.device, dtype=torch.long
        )
        distances = torch.cdist(embeddings, prototypes)
        logits = -distances
        return F.cross_entropy(logits, mapped_labels)

    def fit(self, X, y, epochs=200, batch_size=32, lr=0.001, patience=20):
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._build_encoder().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(epochs):
            indices = torch.randperm(len(X_tensor))
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_tensor), batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]

                optimizer.zero_grad()
                embeddings = self.model(X_batch)
                with torch.no_grad():
                    all_embeddings = self.model(X_tensor)
                prototypes = self._compute_prototypes(all_embeddings, y_tensor)
                loss = self._prototypical_loss(embeddings, y_batch, prototypes)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            scheduler.step()

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        with torch.no_grad():
            all_embeddings = self.model(X_tensor)
        self.prototypes = self._compute_prototypes(all_embeddings, y_tensor)
        self.unique_labels_ = torch.unique(y_tensor).cpu().numpy()
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            embeddings = self.model(X_tensor)
            distances = torch.cdist(embeddings, self.prototypes)
            pred_idx = torch.argmin(distances, dim=1).cpu().numpy()
        return self.unique_labels_[pred_idx]

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            embeddings = self.model(X_tensor)
            distances = torch.cdist(embeddings, self.prototypes)
            proba = torch.softmax(-distances, dim=1).cpu().numpy()
        return proba


# ──────────────────────────────────────────────────────────────
#  ModelWrapper：统一 sklearn 接口
# ──────────────────────────────────────────────────────────────

class ModelWrapper(BaseEstimator, ClassifierMixin):
    """
    模型包装器，使所有模型兼容 sklearn 的 StackingClassifier 接口。
    支持 sklearn 模型、LightGBM、CatBoost 和 PyTorch 模型。
    继承 BaseEstimator 和 ClassifierMixin 以完全兼容 sklearn 接口。
    """
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.n_classes_ = None
        self.classes_ = None

    def get_params(self, deep=True):
        params = {'model_name': self.model_name}
        if deep:
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier
            weight = {0: 1, 1: 1, 2: 1}
            if self.model_name == 'SVM':
                params['model'] = SVC(C=2, kernel='rbf', class_weight=weight,
                                      probability=True, random_state=42)
            elif self.model_name == 'RandomForest':
                params['model'] = RandomForestClassifier(
                    n_estimators=30, max_depth=12, min_samples_leaf=0.015,
                    min_samples_split=0.03, max_features='sqrt',
                    class_weight=weight, bootstrap=True, random_state=100)
            elif self.model_name == 'DecisionTree':
                params['model'] = DecisionTreeClassifier(
                    max_depth=10, min_samples_leaf=0.01, min_samples_split=0.02,
                    criterion='entropy', splitter='random',
                    class_weight=weight, random_state=42)
            elif self.model_name == 'LightGBM':
                params['model'] = LightGBMClassifier()
            elif self.model_name == 'CatBoost':
                params['model'] = CatBoostClassifier()
            elif self.model_name == 'PrototypicalNet':
                input_dim = getattr(self.model, 'input_dim', None)
                if input_dim is None:
                    try:
                        if hasattr(self.model, 'model') and hasattr(self.model.model, '__len__'):
                            first_layer = (self.model.model[0]
                                           if isinstance(self.model.model, (list, tuple))
                                           else list(self.model.model.children())[0])
                            input_dim = first_layer.in_features
                        else:
                            input_dim = 32
                    except Exception:
                        input_dim = 32
                params['model'] = PrototypicalNetwork(input_dim=input_dim)
            else:
                params['model'] = self.model
        else:
            params['model'] = self.model
        return params

    def set_params(self, **params):
        if 'model' in params:
            self.model = params['model']
        if 'model_name' in params:
            self.model_name = params['model_name']
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.model_name in ['LightGBM', 'CatBoost']:
            if hasattr(self.model, 'fit'):
                self.model.fit(X, y)
        elif self.model_name == 'PrototypicalNet':
            if hasattr(self.model, 'fit'):
                try:
                    self.model.fit(X, y, epochs=20)
                except TypeError:
                    self.model.fit(X, y)
        else:
            if hasattr(self.model, 'fit'):
                self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
            if proba.shape[1] != self.n_classes_:
                if proba.shape[1] > self.n_classes_:
                    proba = proba[:, :self.n_classes_]
                else:
                    expanded = np.zeros((proba.shape[0], self.n_classes_))
                    for i, cls in enumerate(self.model.classes_):
                        if cls in self.classes_:
                            idx = np.where(self.classes_ == cls)[0][0]
                            expanded[:, idx] = proba[:, i]
                    proba = expanded
            return proba
        elif self.model_name == 'PrototypicalNet':
            with torch.no_grad():
                logits = self.model.forward(torch.FloatTensor(X))
                proba = torch.softmax(logits, dim=1).numpy()
            return proba
        elif hasattr(self.model, 'predict'):
            pred = self.model.predict(X)
            proba = np.zeros((len(pred), self.n_classes_))
            for i, cls in enumerate(self.classes_):
                mask = (pred == cls)
                if mask.any():
                    proba[mask, i] = 1.0
            return proba
        else:
            return np.ones((X.shape[0], self.n_classes_)) / self.n_classes_

    def predict(self, X):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
