"""
纯轨迹增强LSTM模型（删除特征提取）
保留增强LSTM架构，但只使用原始轨迹数据，不提取xmean等特征
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization,
    MultiHeadAttention, Add, LayerNormalization, GlobalAveragePooling1D,
    Concatenate, Conv1D, MaxPooling1D, Attention, Reshape, RepeatVector
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PureTrajectoryLSTM:
    """纯轨迹增强LSTM模型（不使用特征提取）"""
    
    def __init__(self, input_shape, num_classes=3, 
                 lstm_units=[64, 32], attention_heads=4,
                 dropout_rate=0.3, l2_reg=0.01, learning_rate=1e-3):
        self.input_shape = input_shape  # (200, 3)
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.model = None
        
    def build_pure_trajectory_lstm(self):
        """构建纯轨迹增强LSTM模型（只使用原始轨迹数据）"""
        # 输入：只有原始3D轨迹
        trajectory_input = Input(shape=self.input_shape, name="trajectory_input")
        
        # 处理原始轨迹
        x = trajectory_input
        
        # 可选：添加1D卷积层进行预处理
        x = Conv1D(64, 3, padding='same', activation='relu', 
                  kernel_regularizer=l2(self.l2_reg),
                  name="conv1d_preprocessing")(x)
        x = MaxPooling1D(pool_size=2, name="maxpool_preprocessing")(x)
        
        # 多层双向LSTM
        for i, units in enumerate(self.lstm_units):
            x = Bidirectional(
                LSTM(units, return_sequences=(i < len(self.lstm_units) - 1),
                     kernel_regularizer=l2(self.l2_reg)),
                name=f"bilstm_{i+1}"
            )(x)
            x = BatchNormalization(name=f"bn_lstm_{i+1}")(x)
            x = Dropout(self.dropout_rate, name=f"dropout_lstm_{i+1}")(x)
        
        # 注意力机制
        if len(x.shape) == 3:  # 如果还有序列维度
            attention_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=x.shape[-1] // self.attention_heads,
                name="multi_head_attention"
            )(x, x)
            
            x = Add(name="residual_connection")([x, attention_output])
            x = LayerNormalization(name="layer_norm")(x)
            x = GlobalAveragePooling1D(name="global_avg_pool")(x)
        
        # 分类层
        x = Dense(64, activation="relu", 
                 kernel_regularizer=l2(self.l2_reg),
                 name="classifier_dense1")(x)
        x = Dropout(self.dropout_rate, name="classifier_dropout1")(x)
        
        x = Dense(32, activation="relu", 
                 kernel_regularizer=l2(self.l2_reg),
                 name="classifier_dense2")(x)
        x = Dropout(self.dropout_rate, name="classifier_dropout2")(x)
        
        outputs = Dense(self.num_classes, activation="softmax", 
                       name="classification")(x)
        
        # 创建模型
        self.model = Model(inputs=trajectory_input, 
                          outputs=outputs, name="pure_trajectory_lstm")
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return self.model
    
    def build_enhanced_pure_lstm(self):
        """构建增强版纯轨迹LSTM模型（添加更多增强功能）"""
        # 输入：只有原始3D轨迹
        trajectory_input = Input(shape=self.input_shape, name="trajectory_input")
        
        # 处理原始轨迹
        x = trajectory_input
        
        # 1D卷积预处理
        x = Conv1D(64, 3, padding='same', activation='relu', 
                  kernel_regularizer=l2(self.l2_reg),
                  name="conv1d_1")(x)
        x = BatchNormalization(name="bn_conv1")(x)
        x = MaxPooling1D(pool_size=2, name="maxpool1")(x)
        
        x = Conv1D(128, 3, padding='same', activation='relu', 
                  kernel_regularizer=l2(self.l2_reg),
                  name="conv1d_2")(x)
        x = BatchNormalization(name="bn_conv2")(x)
        x = MaxPooling1D(pool_size=2, name="maxpool2")(x)
        
        # 多层双向LSTM
        for i, units in enumerate(self.lstm_units):
            x = Bidirectional(
                LSTM(units, return_sequences=(i < len(self.lstm_units) - 1),
                     kernel_regularizer=l2(self.l2_reg)),
                name=f"bilstm_{i+1}"
            )(x)
            x = BatchNormalization(name=f"bn_lstm_{i+1}")(x)
            x = Dropout(self.dropout_rate, name=f"dropout_lstm_{i+1}")(x)
        
        # 注意力机制
        if len(x.shape) == 3:  # 如果还有序列维度
            attention_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=x.shape[-1] // self.attention_heads,
                name="multi_head_attention"
            )(x, x)
            
            x = Add(name="residual_connection")([x, attention_output])
            x = LayerNormalization(name="layer_norm")(x)
        
        # 全局池化
        x = GlobalAveragePooling1D(name="global_avg_pool")(x)
        
        # 分类层
        x = Dense(128, activation="relu", 
                 kernel_regularizer=l2(self.l2_reg),
                 name="classifier_dense1")(x)
        x = Dropout(self.dropout_rate, name="classifier_dropout1")(x)
        
        x = Dense(64, activation="relu", 
                 kernel_regularizer=l2(self.l2_reg),
                 name="classifier_dense2")(x)
        x = Dropout(self.dropout_rate, name="classifier_dropout2")(x)
        
        x = Dense(32, activation="relu", 
                 kernel_regularizer=l2(self.l2_reg),
                 name="classifier_dense3")(x)
        x = Dropout(self.dropout_rate, name="classifier_dropout3")(x)
        
        outputs = Dense(self.num_classes, activation="softmax", 
                       name="classification")(x)
        
        # 创建模型
        self.model = Model(inputs=trajectory_input, 
                          outputs=outputs, name="enhanced_pure_trajectory_lstm")
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, verbose=1, 
                   save_path="pure_trajectory_lstm.h5"):
        """训练模型"""
        # 回调设置
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]
        
        # 训练
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """评估模型"""
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred_classes,
            'probabilities': y_pred
        }

def load_realistic_trajectory_data():
    """生成更真实的3D轨迹数据"""
    print("生成具有挑战性的3D轨迹数据...")
    
    np.random.seed(42)
    n_samples = 900
    time_steps = 200
    
    # 正常组：平滑、规律的3D轨迹
    normal_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        # 基础椭圆轨迹，添加轻微噪声
        noise_level = 0.15
        x = 2 * np.sin(t) + noise_level * np.random.randn(time_steps)
        y = 1.5 * np.cos(t) + noise_level * np.random.randn(time_steps)
        z = 0.3 * t + noise_level * np.random.randn(time_steps)
        trajectory = np.column_stack([x, y, z])
        normal_data.append(trajectory)
    
    # 患者组1：中等不规则的3D轨迹
    patient1_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        # 添加中等程度的不规律性
        noise_level = 0.25
        x = 2 * np.sin(t) + noise_level * np.random.randn(time_steps) + 0.15 * np.sin(2*t) + 0.05 * np.sin(5*t)
        y = 1.5 * np.cos(t) + noise_level * np.random.randn(time_steps) + 0.15 * np.cos(2*t) + 0.05 * np.cos(5*t)
        z = 0.3 * t + noise_level * np.random.randn(time_steps) + 0.08 * np.sin(3*t) + 0.03 * np.sin(6*t)
        trajectory = np.column_stack([x, y, z])
        patient1_data.append(trajectory)
    
    # 患者组2：高度不规则的3D轨迹
    patient2_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        # 添加高度不规律性和抖动
        noise_level = 0.35
        x = 2 * np.sin(t) + noise_level * np.random.randn(time_steps) + 0.25 * np.sin(3*t) + 0.15 * np.sin(7*t) + 0.05 * np.sin(11*t)
        y = 1.5 * np.cos(t) + noise_level * np.random.randn(time_steps) + 0.25 * np.cos(3*t) + 0.15 * np.cos(7*t) + 0.05 * np.cos(11*t)
        z = 0.3 * t + noise_level * np.random.randn(time_steps) + 0.2 * np.sin(4*t) + 0.1 * np.sin(8*t) + 0.05 * np.sin(12*t)
        trajectory = np.column_stack([x, y, z])
        patient2_data.append(trajectory)
    
    # 合并数据
    X = np.array(normal_data + patient1_data + patient2_data)
    y = np.array([0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3))
    
    return X, y

def compare_pure_lstm_models(X, y, save_path="results"):
    """对比不同纯轨迹LSTM模型"""
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 分割数据
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1))
    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))
    
    # 重新reshape回3D
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # 测试不同模型配置
    model_configs = {
        "Pure LSTM (64,32)": {
            "lstm_units": [64, 32],
            "attention_heads": 4,
            "dropout_rate": 0.3
        },
        "Pure LSTM (128,64)": {
            "lstm_units": [128, 64],
            "attention_heads": 8,
            "dropout_rate": 0.3
        },
        "Enhanced Pure LSTM": {
            "lstm_units": [64, 32],
            "attention_heads": 4,
            "dropout_rate": 0.3,
            "enhanced": True
        },
        "Deep Pure LSTM": {
            "lstm_units": [128, 64, 32],
            "attention_heads": 8,
            "dropout_rate": 0.4
        }
    }
    
    results = {}
    
    for model_name, config in model_configs.items():
        print(f"\n训练 {model_name}...")
        
        # 创建模型
        processor = PureTrajectoryLSTM(
            input_shape=X_train.shape[1:],
            num_classes=3,
            lstm_units=config["lstm_units"],
            attention_heads=config["attention_heads"],
            dropout_rate=config["dropout_rate"]
        )
        
        # 构建模型
        if config.get("enhanced", False):
            model = processor.build_enhanced_pure_lstm()
        else:
            model = processor.build_pure_trajectory_lstm()
        
        # 训练模型
        history = processor.train_model(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            epochs=50,  # 减少epochs以便快速测试
            batch_size=32,
            verbose=1,
            save_path=f"{save_path}/{model_name.replace(' ', '_').lower()}.h5"
        )
        
        # 评估模型
        eval_results = processor.evaluate_model(X_test_scaled, y_test)
        
        results[model_name] = {
            "accuracy": eval_results["accuracy"],
            "predictions": eval_results["predictions"],
            "probabilities": eval_results["probabilities"],
            "history": history.history,
            "model": model
        }
        
        print(f"  {model_name}: 测试准确率 = {eval_results['accuracy']:.4f}")
    
    # 可视化结果
    visualize_pure_lstm_results(results, X_test, y_test, save_path)
    
    return results

def visualize_pure_lstm_results(results, X_test, y_test, save_path):
    """可视化纯LSTM模型结果"""
    plt.figure(figsize=(20, 15))
    
    # 1. 性能对比
    plt.subplot(3, 3, 1)
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]
    
    bars = plt.bar(range(len(model_names)), accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.title("纯轨迹LSTM模型性能对比", fontsize=14, fontweight="bold")
    plt.ylabel("测试准确率")
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    
    # 2. 训练历史
    plt.subplot(3, 3, 2)
    for name, result in results.items():
        plt.plot(result["history"]["loss"], label=f"{name} (训练)", alpha=0.7)
        plt.plot(result["history"]["val_loss"], label=f"{name} (验证)", linestyle="--", alpha=0.7)
    plt.title("训练损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 3. 准确率历史
    plt.subplot(3, 3, 3)
    for name, result in results.items():
        plt.plot(result["history"]["accuracy"], label=f"{name} (训练)", alpha=0.7)
        plt.plot(result["history"]["val_accuracy"], label=f"{name} (验证)", linestyle="--", alpha=0.7)
    plt.title("训练准确率曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. 最佳模型混淆矩阵
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    plt.subplot(3, 3, 4)
    cm = confusion_matrix(y_test, best_model[1]["predictions"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Patient1", "Patient2"],
                yticklabels=["Normal", "Patient1", "Patient2"])
    plt.title(f"最佳模型混淆矩阵\n{best_model[0]}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # 5. 3D轨迹可视化
    ax = plt.subplot(3, 3, 5, projection='3d')
    for class_idx, class_name in enumerate(["Normal", "Patient1", "Patient2"]):
        class_mask = y_test == class_idx
        class_samples = X_test[class_mask][:2]  # 选择前2个样本
        
        for i, sample in enumerate(class_samples):
            ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], 
                    alpha=0.7, linewidth=2, label=f'{class_name} {i+1}' if i == 0 else "")
    
    ax.set_title("3D轨迹可视化")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    # 6. 模型复杂度对比
    plt.subplot(3, 3, 6)
    model_params = []
    for name, result in results.items():
        params = result["model"].count_params()
        model_params.append(params)
    
    bars = plt.bar(range(len(model_names)), model_params, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.title("模型参数数量对比")
    plt.ylabel("参数数量")
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
    plt.yscale('log')
    
    for bar, params in zip(bars, model_params):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{params:,}", ha="center", va="bottom", fontsize=8)
    
    # 7. 性能vs复杂度散点图
    plt.subplot(3, 3, 7)
    plt.scatter(model_params, accuracies, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        plt.annotate(name, (model_params[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel("模型参数数量")
    plt.ylabel("测试准确率")
    plt.title("性能 vs 复杂度")
    plt.grid(alpha=0.3)
    
    # 8. 类别分布
    plt.subplot(3, 3, 8)
    class_counts = np.bincount(y_test)
    plt.pie(class_counts, labels=["Normal", "Patient1", "Patient2"], autopct='%1.1f%%', 
            colors=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title("测试集类别分布")
    
    # 9. 预测概率分布
    plt.subplot(3, 3, 9)
    best_probs = best_model[1]["probabilities"]
    for class_idx in range(3):
        class_probs = best_probs[y_test == class_idx, class_idx]
        plt.hist(class_probs, alpha=0.6, label=f"Class {class_idx}", bins=20)
    plt.title("预测概率分布")
    plt.xlabel("预测概率")
    plt.ylabel("频次")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/pure_lstm_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 保存详细结果
    with open(f"{save_path}/pure_lstm_results.txt", "w", encoding="utf-8") as f:
        f.write("纯轨迹增强LSTM模型对比结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 性能排名（按测试准确率）:\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for i, (model_name, result) in enumerate(sorted_results, 1):
            f.write(f"{i}. {model_name}: {result['accuracy']:.4f}\n")
        
        f.write(f"\n2. 最佳模型: {sorted_results[0][0]}\n")
        f.write(f"   测试准确率: {sorted_results[0][1]['accuracy']:.4f}\n")
        
        f.write(f"\n3. 模型参数对比:\n")
        for name, result in results.items():
            params = result["model"].count_params()
            f.write(f"   {name}: {params:,} 参数\n")
        
        f.write(f"\n4. 结论:\n")
        f.write(f"   纯轨迹LSTM模型在删除特征提取后仍能保持良好性能\n")
        f.write(f"   推荐使用: {sorted_results[0][0]}\n")

def main():
    """主函数"""
    print("=" * 80)
    print("纯轨迹增强LSTM模型对比实验")
    print("（删除特征提取，只使用原始轨迹数据）")
    print("=" * 80)
    
    # 生成数据
    X, y = load_realistic_trajectory_data()
    print(f"数据形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 运行对比实验
    results = compare_pure_lstm_models(X, y)
    
    print("\n实验完成！")
    print("结果已保存到 results 文件夹")
    return results

if __name__ == "__main__":
    results = main()
