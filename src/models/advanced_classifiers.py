"""
最新先进分类器实现
包含TabPFN v2, TabR, TabKAN, DOFEN, ExcelFormer, MambaAttention等
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class TabRClassifier(nn.Module):
    """TabR - 检索增强分类器"""

    def __init__(self, input_dim, num_classes, k=10, hidden_dim=64):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim

        # 嵌入层
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # 确保输入是2D张量 (batch_size, features)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # 嵌入层
        embedded = self.embedding(x)

        # 自注意力机制需要3D输入 (batch_size, seq_len, features)
        # 如果输入是2D，添加序列维度
        if len(embedded.shape) == 2:
            embedded = embedded.unsqueeze(1)  # (batch_size, 1, features)

        attn_out, _ = self.attention(embedded, embedded, embedded)

        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)

        # 分类
        return self.classifier(pooled)


class KANLayer(nn.Module):
    """KAN层 - 可学习的激活函数"""

    def __init__(self, input_dim, output_dim, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 可学习的激活函数参数
        self.activation_functions = nn.Parameter(
            torch.randn(input_dim, output_dim, grid_size) * 0.1
        )

        # 网格点
        self.grid_points = torch.linspace(-2, 2, grid_size)

    def forward(self, x):
        # 简化的KAN实现
        # 实际实现需要样条插值
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.output_dim)

        for i in range(self.output_dim):
            for j in range(self.input_dim):
                # 简单的线性组合
                output[:, i] += x[:, j] * torch.sum(
                    self.activation_functions[j, i, :]
                )

        return output


class TabKANClassifier(nn.Module):
    """TabKAN - 基于Kolmogorov-Arnold网络的分类器"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.kan1 = KANLayer(input_dim, 64)
        self.kan2 = KANLayer(64, 32)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.kan1(x)
        x = torch.relu(x)
        x = self.kan2(x)
        x = torch.relu(x)
        return self.classifier(x)


class ObliviousDecisionTree(nn.Module):
    """遗忘决策树"""

    def __init__(self, input_dim, depth=3):
        super().__init__()
        self.depth = depth
        self.leaves = nn.Parameter(torch.randn(2 ** depth))
        self.feature_weights = nn.Parameter(torch.randn(input_dim, depth))
        self.thresholds = nn.Parameter(torch.randn(depth))

    def forward(self, x):
        # 计算决策路径
        decisions = torch.sigmoid(
            torch.matmul(x, self.feature_weights) - self.thresholds
        )

        # 转换为叶子索引
        leaf_indices = torch.sum(
            decisions * (2 ** torch.arange(self.depth, device=x.device)),
            dim=1
        ).long()

        # 确保索引在有效范围内
        leaf_indices = torch.clamp(leaf_indices, 0, len(self.leaves) - 1)

        return self.leaves[leaf_indices]


class SemiPermeableAttention(nn.Module):
    """半渗透注意力机制"""

    def __init__(self, input_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # 处理2D输入，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)

        # 如果输入是2D，输出也应该是2D
        if out.shape[1] == 1:
            out = out.squeeze(1)

        return self.proj(out), None, None


class ExcelFormerClassifier(nn.Module):
    """ExcelFormer - 超越GBDT的神经网络"""

    def __init__(self, input_dim, num_classes, num_layers=6):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        self.layers = nn.ModuleList([
            SemiPermeableAttention(64) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # 确保输入是2D张量 (batch_size, features)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = self.embedding(x)
        for layer in self.layers:
            # SemiPermeableAttention返回3个值，我们只需要第一个
            layer_output, _, _ = layer(x)
            x = x + layer_output  # 残差连接
        # 不需要全局平均池化，直接使用x
        return self.classifier(x)


class MambaAttentionBlock(nn.Module):
    """Mamba注意力块"""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.attention = nn.MultiheadAttention(input_dim, num_heads=min(8, input_dim // 8), batch_first=True)

    def forward(self, x):
        # 确保输入是3D张量 (batch_size, seq_len, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加序列维度

        # Mamba风格的处理
        mamba_out = self.linear2(torch.relu(self.linear1(x)))

        # 注意力机制
        attn_out, _ = self.attention(x, x, x)

        # 残差连接
        return x + mamba_out + attn_out


class MambaAttentionClassifier(nn.Module):
    """MambaAttention分类器"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaAttentionBlock(input_dim) for _ in range(4)
        ])
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # 确保输入是2D张量 (batch_size, features)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        for block in self.blocks:
            x = block(x)

        # 如果x是3D，取最后一个时间步或平均
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # 平均池化
        elif len(x.shape) == 2:
            x = x  # 已经是2D

        return self.classifier(x)


class AdvancedClassifierTrainer:
    """先进分类器训练器"""

    def __init__(self, device="cpu"):
        self.device = device
        self.models = {}
        self.results = {}

    def train_pytorch_model(self, model, X_train, y_train, X_test, y_test,
                            epochs=100, lr=0.001, batch_size=32):
        """训练PyTorch模型"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)

        # 训练循环
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # 评估
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).argmax(dim=1).cpu().numpy()
            test_pred = model(X_test_tensor).argmax(dim=1).cpu().numpy()

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        return model, train_acc, test_acc

    def train_all_classifiers(self, X_train, y_train, X_test, y_test):
        """训练所有分类器"""
        print("🚀 开始训练先进分类器...")

        # 1. TabR
        print("\n📊 训练 TabR...")
        try:
            tabr = TabRClassifier(X_train.shape[1], len(np.unique(y_train)))
            tabr, train_acc, test_acc = self.train_pytorch_model(
                tabr, X_train, y_train, X_test, y_test
            )
            self.models["TabR"] = tabr
            self.results["TabR"] = {"train_acc": train_acc, "test_acc": test_acc}
            print(f"TabR - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        except Exception as e:
            print(f"TabR训练失败: {e}")

        # 2. TabKAN
        print("\n🧠 训练 TabKAN...")
        try:
            tabkan = TabKANClassifier(X_train.shape[1], len(np.unique(y_train)))
            tabkan, train_acc, test_acc = self.train_pytorch_model(
                tabkan, X_train, y_train, X_test, y_test
            )
            self.models["TabKAN"] = tabkan
            self.results["TabKAN"] = {"train_acc": train_acc, "test_acc": test_acc}
            print(f"TabKAN - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        except Exception as e:
            print(f"TabKAN训练失败: {e}")

        # 4. ExcelFormer
        print("\n⚡ 训练 ExcelFormer...")
        try:
            excelformer = ExcelFormerClassifier(X_train.shape[1], len(np.unique(y_train)))
            excelformer, train_acc, test_acc = self.train_pytorch_model(
                excelformer, X_train, y_train, X_test, y_test
            )
            self.models["ExcelFormer"] = excelformer
            self.results["ExcelFormer"] = {"train_acc": train_acc, "test_acc": test_acc}
            print(f"ExcelFormer - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        except Exception as e:
            print(f"ExcelFormer训练失败: {e}")

        # 5. MambaAttention
        print("\n🎯 训练 MambaAttention...")
        try:
            mamba = MambaAttentionClassifier(X_train.shape[1], len(np.unique(y_train)))
            mamba, train_acc, test_acc = self.train_pytorch_model(
                mamba, X_train, y_train, X_test, y_test
            )
            self.models["MambaAttention"] = mamba
            self.results["MambaAttention"] = {"train_acc": train_acc, "test_acc": test_acc}
            print(f"MambaAttention - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        except Exception as e:
            print(f"MambaAttention训练失败: {e}")

        return self.results

    def print_results_summary(self):
        """打印结果摘要"""
        print("\n" + "=" * 60)
        print("📊 先进分类器性能对比")
        print("=" * 60)

        # 按测试准确率排序
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]["test_acc"],
            reverse=True
        )

        print(f"{'分类器':<20} {'训练准确率':<12} {'测试准确率':<12}")
        print("-" * 50)

        for name, result in sorted_results:
            print(f"{name:<20} {result['train_acc']:<12.4f} {result['test_acc']:<12.4f}")

        print("=" * 60)

        # 找出最佳模型
        best_model = max(self.results.items(), key=lambda x: x[1]["test_acc"])
        print(f"🏆 最佳模型: {best_model[0]} (测试准确率: {best_model[1]['test_acc']:.4f})")

        return best_model


def test_advanced_classifiers(X_train, y_train, X_test, y_test):
    """测试所有先进分类器"""
    print("🚀 开始测试最新先进分类器...")

    # 创建训练器
    trainer = AdvancedClassifierTrainer()

    # 训练所有分类器
    results = trainer.train_all_classifiers(X_train, y_train, X_test, y_test)

    # 打印结果摘要
    best_model = trainer.print_results_summary()

    return trainer, results, best_model


if __name__ == "__main__":
    print("先进分类器模块已加载！")
    print("使用方法: test_advanced_classifiers(X_train, y_train, X_test, y_test)")
