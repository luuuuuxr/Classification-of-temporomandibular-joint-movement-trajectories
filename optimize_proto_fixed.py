# 优化PrototypicalNet脚本
import re

# 读取文件内容
with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# 新的优化版PrototypicalNet类
new_proto_class = """class PrototypicalNetwork:
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        self.prototypes = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        # 计算每个类别的原型
        unique_labels = torch.unique(labels)
        prototypes = []
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                prototype = embeddings[mask].mean(dim=0)
                prototypes.append(prototype)
        return torch.stack(prototypes) if prototypes else torch.empty(0, embeddings.size(1))
    
    def _prototypical_loss(self, embeddings, labels, prototypes):
        # 计算原型网络损失
        if len(prototypes) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # 计算到每个原型的距离
        distances = torch.cdist(embeddings, prototypes)
        # 转换为logits（负距离）
        logits = -distances
        return F.cross_entropy(logits, labels)
        
    def fit(self, X, y, epochs=200, batch_size=32, lr=0.001, patience=20):
        # 训练原型网络
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._build_encoder().to(self.device)
        
        # 数据分割
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # 优化器和调度器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=False
        )
        
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # 批次训练
            for i in range(0, len(X_train_tensor), batch_size):
                batch_end = min(i + batch_size, len(X_train_tensor))
                batch_x = X_train_tensor[i:batch_end]
                batch_y = y_train_tensor[i:batch_end]
                
                optimizer.zero_grad()
                embeddings = self.model(batch_x)
                prototypes = self._compute_prototypes(embeddings, batch_y)
                loss = self._prototypical_loss(embeddings, batch_y, prototypes)
                
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_embeddings = self.model(X_val_tensor)
                val_prototypes = self._compute_prototypes(val_embeddings, y_val_tensor)
                val_loss = self._prototypical_loss(val_embeddings, y_val_tensor, val_prototypes)
            self.model.train()
            
            avg_train_loss = total_loss / num_batches
            scheduler.step(val_loss)
            
            # 早停机制
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 计算最终原型
        self.model.eval()
        with torch.no_grad():
            all_embeddings = self.model(torch.FloatTensor(X_scaled).to(self.device))
            all_labels = torch.LongTensor(y).to(self.device)
            self.prototypes = self._compute_prototypes(all_embeddings, all_labels)
        
        return self
        
    def predict(self, X):
        # 预测
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(X_tensor)
            if len(self.prototypes) > 0:
                distances = torch.cdist(embeddings, self.prototypes)
                predictions = torch.argmin(distances, dim=1)
            else:
                # 如果没有原型，返回随机预测
                predictions = torch.randint(0, 3, (len(X),), device=self.device)
        
        return predictions.cpu().numpy()
        
    def set_training_data(self, X, y):
        self.X_train = X
        self.y_train = y"""

# 使用正则表达式替换整个PrototypicalNetwork类
pattern = r"class PrototypicalNetwork:.*?(?=\n\n# ========== 新分类器实现结束 ==========)"
replacement = new_proto_class + "\n\n# ========== 新分类器实现结束 =========="

if re.search(pattern, content, re.DOTALL):
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ PrototypicalNetwork类已成功优化！")
else:
    print("❌ 未找到PrototypicalNetwork类定义")
    # 显示当前类的位置
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "class PrototypicalNetwork:" in line:
            print(f"第{i+1}行: {line}")
            break
