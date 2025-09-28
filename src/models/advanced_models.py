import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# 尝试导入LightGBM和CatBoost
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")

class LightGBMClassifier:
    """LightGBM分类器包装"""
    
    def __init__(self, **params):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1
        }
        default_params.update(params)
        
        self.model = lgb.LGBMClassifier(**default_params)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

class CatBoostClassifier:
    """CatBoost分类器包装"""
    
    def __init__(self, **params):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")
        
        default_params = {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "random_seed": 42,
            "verbose": False
        }
        default_params.update(params)
        
        self.model = cb.CatBoostClassifier(**default_params)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

class PrototypicalNetwork(nn.Module):
    """原型网络实现"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super(PrototypicalNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.scaler = StandardScaler()
        
    def forward(self, x):
        return self.encoder(x)
    
    def fit(self, X, y, epochs=100, lr=0.001, batch_size=32):
        """训练原型网络"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        self.train()
        for epoch in range(epochs):
            # 随机采样
            indices = torch.randperm(len(X_tensor))
            total_loss = 0
            
            for i in range(0, len(X_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_x = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]
                
                optimizer.zero_grad()
                features = self.forward(batch_x)
                
                # 计算原型
                prototypes = self.compute_prototypes(features, batch_y)
                
                # 计算距离和损失
                distances = self.compute_distances(features, prototypes)
                loss = criterion(-distances, batch_y)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X_tensor)*batch_size:.4f}")
        
        return self
    
    def compute_prototypes(self, features, labels):
        """计算每个类别的原型"""
        unique_labels = torch.unique(labels)
        prototypes = []
        
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                prototype = features[mask].mean(dim=0)
                prototypes.append(prototype)
            else:
                prototypes.append(torch.zeros_like(features[0]))
        
        return torch.stack(prototypes)
    
    def compute_distances(self, features, prototypes):
        """计算特征到原型的距离"""
        distances = torch.cdist(features, prototypes, p=2)
        return -distances  # 返回负距离（相似度）
    
    def predict(self, X):
        """预测"""
        device = next(self.parameters()).device
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        self.eval()
        with torch.no_grad():
            features = self.forward(X_tensor)
            
            # 使用训练数据计算原型
            train_features = self.forward(torch.FloatTensor(self.scaler.transform(self.train_X)).to(device))
            prototypes = self.compute_prototypes(train_features, torch.LongTensor(self.train_y).to(device))
            
            distances = self.compute_distances(features, prototypes)
            predictions = torch.argmax(distances, dim=1)
            
        return predictions.cpu().numpy()
    
    def set_training_data(self, X, y):
        """设置训练数据用于计算原型"""
        self.train_X = X
        self.train_y = y

def test_advanced_models():
    """测试所有先进模型"""
    print("测试先进模型...")
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 64
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    
    # 分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = {}
    
    # 测试LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n测试LightGBM...")
        lgb_model = LightGBMClassifier()
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        lgb_f1 = f1_score(y_test, lgb_pred, average="macro")
        results["LightGBM"] = (lgb_acc, lgb_f1)
        print(f"LightGBM: 准确率={lgb_acc:.4f}, F1={lgb_f1:.4f}")
    
    # 测试CatBoost
    if CATBOOST_AVAILABLE:
        print("\n测试CatBoost...")
        cb_model = CatBoostClassifier()
        cb_model.fit(X_train, y_train)
        cb_pred = cb_model.predict(X_test)
        cb_acc = accuracy_score(y_test, cb_pred)
        cb_f1 = f1_score(y_test, cb_pred, average="macro")
        results["CatBoost"] = (cb_acc, cb_f1)
        print(f"CatBoost: 准确率={cb_acc:.4f}, F1={cb_f1:.4f}")
    
    # 测试Prototypical Network
    print("\n测试Prototypical Network...")
    try:
        proto_model = PrototypicalNetwork(n_features)
        proto_model.set_training_data(X_train, y_train)
        proto_model.fit(X_train, y_train, epochs=50)
        proto_pred = proto_model.predict(X_test)
        proto_acc = accuracy_score(y_test, proto_pred)
        proto_f1 = f1_score(y_test, proto_pred, average="macro")
        results["Prototypical Network"] = (proto_acc, proto_f1)
        print(f"Prototypical Network: 准确率={proto_acc:.4f}, F1={proto_f1:.4f}")
    except Exception as e:
        print(f"Prototypical Network 训练失败: {e}")

    # 显示结果
    print("\n所有模型结果:")
    for name, (acc, f1) in results.items():
        print(f"{name}: 准确率={acc:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    test_advanced_models()
