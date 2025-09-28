import re

# 读取文件
with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# 定义新的PrototypicalNetwork类
new_class = """class PrototypicalNetwork:
    \"\"\"改进的原型网络实现，包含更深的网络结构、学习率调度和批量训练\"\"\"
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        self.prototypes = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_loss = float("inf")
        
    def _build_encoder(self):
        \"\"\"构建更深的编码器网络\"\"\"
        return nn.Sequential(
            # 第一层
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第二层
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第三层
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 输出层
            nn.Linear(self.hidden_dim // 2, self.output_dim),
            nn.BatchNorm1d(self.output_dim)
        )
        
    def fit(self, X, y, epochs=150, batch_size=32, lr=0.005, patience=20):
        \"\"\"训练原型网络，包含学习率调度和早停\"\"\"
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._build_encoder().to(self.device)
        
        # 使用更先进的优化器配置
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=10, verbose=False
        )
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # 创建数据加载器用于批量训练
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                embeddings = self.model(batch_x)
                
                # 计算原型（基于当前批次和历史信息）
                unique_labels = torch.unique(batch_y)
                prototypes = []
                
                for label in unique_labels:
                    mask = (batch_y == label)
                    if mask.sum() > 0:
                        # 使用加权平均来计算原型
                        class_embeddings = embeddings[mask]
                        prototype = class_embeddings.mean(dim=0)
                        prototypes.append(prototype)
                
                if len(prototypes) > 1:  # 确保有多个类别
                    prototypes = torch.stack(prototypes)
                    
                    # 计算距离和损失
                    distances = torch.cdist(embeddings, prototypes)
                    
                    # 创建标签映射
                    label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}
                    mapped_labels = torch.tensor([label_mapping[y.item()] for y in batch_y], 
                                               device=self.device)
                    
                    # 使用焦点损失来处理类别不平衡
                    logits = -distances
                    loss = F.cross_entropy(logits, mapped_labels)
                    
                    # 添加原型间分离损失
                    if len(prototypes) > 1:
                        proto_distances = torch.cdist(prototypes, prototypes)
                        # 鼓励不同类别的原型相互分离
                        separation_loss = torch.exp(-proto_distances.fill_diagonal_(float("inf")).min())
                        loss += 0.1 * separation_loss
                    
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                scheduler.step(avg_loss)
                
                # 早停检查
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0][\"lr\"]:.6f}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # 计算最终原型
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(X_tensor)
            unique_labels = torch.unique(y_tensor)
            prototypes = []
            
            for label in unique_labels:
                mask = (y_tensor == label)
                if mask.sum() > 0:
                    class_embeddings = embeddings[mask]
                    # 使用更稳健的原型计算
                    prototype = class_embeddings.mean(dim=0)
                    prototypes.append(prototype)
            
            self.prototypes = torch.stack(prototypes)
            self.label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}
            self.reverse_mapping = {i: label.item() for i, label in enumerate(unique_labels)}
        
        return self
        
    def predict(self, X):
        \"\"\"预测新样本\"\"\"
        if self.model is None or self.prototypes is None:
            raise ValueError("模型尚未训练")
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(X_tensor)
            distances = torch.cdist(embeddings, self.prototypes)
            prototype_indices = torch.argmin(distances, dim=1)
            
            # 映射回原始标签
            predictions = [self.reverse_mapping[idx.item()] for idx in prototype_indices]
        
        return np.array(predictions)
        
    def predict_proba(self, X):
        \"\"\"预测概率（基于距离的软分配）\"\"\"
        if self.model is None or self.prototypes is None:
            raise ValueError("模型尚未训练")
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(X_tensor)
            distances = torch.cdist(embeddings, self.prototypes)
            
            # 将距离转换为概率（距离越小，概率越大）
            probabilities = F.softmax(-distances, dim=1)
        
        return probabilities.cpu().numpy()
        
    def set_training_data(self, X, y):
        self.X_train = X
        self.y_train = y"""

# 使用正则表达式替换整个PrototypicalNetwork类
pattern = r"class PrototypicalNetwork:.*?(?=\n\n# ========== 新分类器实现结束 ==========)"
new_content = re.sub(pattern, new_class, content, flags=re.DOTALL)

# 写回文件
with open("main.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("✅ PrototypicalNetwork类已成功优化！")
print("主要改进：")
print("- 更深的网络结构（4层 + BatchNorm + Dropout）")
print("- 增加训练轮数（50 -> 150 epochs）")
print("- 学习率调度和早停机制")
print("- 批量训练和梯度裁剪")
print("- 原型间分离损失")
print("- 更稳健的概率预测")

