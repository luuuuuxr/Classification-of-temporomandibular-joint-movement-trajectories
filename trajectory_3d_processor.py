"""
3D轨迹数据专用处理模块
针对颞下颌关节3D运动轨迹的优化架构
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, Add, LayerNormalization,
    Concatenate, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os

class Trajectory3DProcessor:
    """3D轨迹数据专用处理器"""
    
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
        
    def extract_3d_spatial_features(self, trajectory):
        """提取3D轨迹的空间几何特征"""
        features = []
        
        # 1. 3D坐标统计特征
        for coord in range(3):  # x, y, z
            coord_data = trajectory[:, coord]
            features.extend([
                np.mean(coord_data),      # 均值
                np.std(coord_data),       # 标准差
                np.ptp(coord_data),       # 极差
                np.median(coord_data),    # 中位数
                np.percentile(coord_data, 25),  # 25分位数
                np.percentile(coord_data, 75),  # 75分位数
            ])
        
        # 2. 运动学特征
        if len(trajectory) > 1:
            velocity = np.diff(trajectory, axis=0)
            speed = np.linalg.norm(velocity, axis=1)
            
            features.extend([
                np.mean(speed),           # 平均速度
                np.std(speed),            # 速度标准差
                np.max(speed),            # 最大速度
                np.min(speed),            # 最小速度
            ])
            
            # 加速度特征
            if len(velocity) > 1:
                acceleration = np.diff(velocity, axis=0)
                accel_magnitude = np.linalg.norm(acceleration, axis=1)
                features.extend([
                    np.mean(accel_magnitude),  # 平均加速度
                    np.std(accel_magnitude),   # 加速度标准差
                ])
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 3. 3D几何特征
        if len(trajectory) > 2:
            # 轨迹总长度
            trajectory_length = np.sum(np.linalg.norm(velocity, axis=1))
            features.append(trajectory_length)
            
            # 直线距离
            straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
            features.append(straight_distance)
            
            # 轨迹复杂度（直线距离/轨迹长度）
            complexity = straight_distance / (trajectory_length + 1e-8)
            features.append(complexity)
            
            # 轨迹弯曲度
            curvature = self._calculate_curvature(trajectory)
            features.append(curvature)
            
            # 3D空间范围
            x_range = np.ptp(trajectory[:, 0])
            y_range = np.ptp(trajectory[:, 1])
            z_range = np.ptp(trajectory[:, 2])
            features.extend([x_range, y_range, z_range])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def _calculate_curvature(self, trajectory):
        """计算轨迹弯曲度"""
        if len(trajectory) < 3:
            return 0.0
        
        # 计算曲率
        first_deriv = np.gradient(trajectory, axis=0)
        second_deriv = np.gradient(first_deriv, axis=0)
        
        # 3D曲率公式
        cross_product = np.cross(first_deriv, second_deriv, axis=1)
        curvature = np.linalg.norm(cross_product, axis=1) / (np.linalg.norm(first_deriv, axis=1) ** 3 + 1e-8)
        
        return np.mean(curvature)
    
    def extract_frequency_features(self, trajectory):
        """提取频域特征"""
        features = []
        
        for coord in range(3):  # x, y, z坐标
            coord_data = trajectory[:, coord]
            
            # FFT变换
            fft_data = fft(coord_data)
            freqs = fftfreq(len(coord_data))
            
            # 只取正频率部分
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft_data[:len(fft_data)//2])
            
            if len(positive_fft) > 0:
                features.extend([
                    np.max(positive_fft),                    # 最大幅值
                    np.mean(positive_fft),                   # 平均幅值
                    np.std(positive_fft),                    # 幅值标准差
                    positive_freqs[np.argmax(positive_fft)], # 主频率
                    np.sum(positive_fft),                    # 总能量
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def preprocess_trajectories(self, X):
        """预处理轨迹数据，提取3D特征"""
        spatial_features = []
        freq_features = []
        
        for trajectory in X:
            # 提取空间特征
            spatial_feat = self.extract_3d_spatial_features(trajectory)
            spatial_features.append(spatial_feat)
            
            # 提取频域特征
            freq_feat = self.extract_frequency_features(trajectory)
            freq_features.append(freq_feat)
        
        # 合并特征
        spatial_features = np.array(spatial_features)
        freq_features = np.array(freq_features)
        combined_features = np.concatenate([spatial_features, freq_features], axis=1)
        
        return combined_features, spatial_features, freq_features
    
    def build_simple_lstm_model(self):
        """构建简化的LSTM模型（只使用原始轨迹）"""
        inputs = Input(shape=self.input_shape, name="trajectory_input")
        
        x = inputs
        
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
        if len(x.shape) == 3:
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
                 name="dense_1")(x)
        x = Dropout(self.dropout_rate, name="dropout_1")(x)
        
        x = Dense(32, activation="relu", 
                 kernel_regularizer=l2(self.l2_reg),
                 name="dense_2")(x)
        x = Dropout(self.dropout_rate, name="dropout_2")(x)
        
        outputs = Dense(self.num_classes, activation="softmax", 
                       name="classification")(x)
        
        # 创建模型
        self.model = Model(inputs=inputs, outputs=outputs, name="simple_3d_lstm")
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return self.model
    
    def build_enhanced_lstm_model(self):
        """构建增强的LSTM模型（原始轨迹 + 3D特征）"""
        # 输入：原始3D轨迹 + 提取的特征
        trajectory_input = Input(shape=self.input_shape, name="trajectory_input")
        spatial_input = Input(shape=(None,), name="spatial_features")  # 动态特征维度
        
        # 处理原始轨迹
        x = trajectory_input
        
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
        
        # 处理空间特征
        spatial_dense = Dense(32, activation="relu", 
                            kernel_regularizer=l2(self.l2_reg),
                            name="spatial_dense")(spatial_input)
        spatial_dense = Dropout(self.dropout_rate, name="spatial_dropout")(spatial_dense)
        
        # 融合特征
        if len(x.shape) == 1:  # 如果x已经是1D
            x = tf.keras.layers.Reshape((1, -1))(x)
        
        # 将空间特征重复到序列长度
        spatial_repeated = tf.keras.layers.RepeatVector(x.shape[1])(spatial_dense)
        
        # 拼接轨迹特征和空间特征
        combined = Concatenate(axis=-1, name="feature_fusion")([x, spatial_repeated])
        
        # 最终处理
        combined = Dense(64, activation="relu", 
                        kernel_regularizer=l2(self.l2_reg),
                        name="fusion_dense")(combined)
        combined = Dropout(self.dropout_rate, name="fusion_dropout")(combined)
        
        # 全局池化
        if len(combined.shape) == 3:
            combined = GlobalAveragePooling1D(name="final_pool")(combined)
        
        # 分类层
        x = Dense(32, activation="relu", 
                 kernel_regularizer=l2(self.l2_reg),
                 name="classifier_dense")(combined)
        x = Dropout(self.dropout_rate, name="classifier_dropout")(x)
        
        outputs = Dense(self.num_classes, activation="softmax", 
                       name="classification")(x)
        
        # 创建模型
        self.model = Model(inputs=[trajectory_input, spatial_input], 
                          outputs=outputs, name="enhanced_3d_lstm")
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1, use_spatial_features=True):
        """训练模型"""
        if use_spatial_features:
            # 提取空间特征
            spatial_features, _, _ = self.preprocess_trajectories(X_train)
            spatial_features_val, _, _ = self.preprocess_trajectories(X_val)
            
            if self.model is None:
                self.build_enhanced_lstm_model()
            
            # 训练
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)
            ]
            
            history = self.model.fit(
                [X_train, spatial_features], y_train,
                validation_data=([X_val, spatial_features_val], y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            if self.model is None:
                self.build_simple_lstm_model()
            
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        
        return history
    
    def predict(self, X, use_spatial_features=True):
        """预测"""
        if use_spatial_features and hasattr(self, "model") and len(self.model.inputs) == 2:
            spatial_features, _, _ = self.preprocess_trajectories(X)
            return self.model.predict([X, spatial_features], verbose=0)
        else:
            return self.model.predict(X, verbose=0)

def compare_3d_architectures(X_train, y_train, X_val, y_val, input_shape, save_path="results"):
    """比较不同3D轨迹处理架构"""
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    results = {}
    histories = {}
    
    # 1. 简单LSTM（无3D特征）
    print("=" * 60)
    print("训练简单LSTM模型（无3D特征）...")
    print("=" * 60)
    
    simple_lstm = Trajectory3DProcessor(input_shape)
    simple_history = simple_lstm.train(X_train, y_train, X_val, y_val, 
                                      epochs=50, batch_size=32, verbose=1, 
                                      use_spatial_features=False)
    results["Simple LSTM"] = max(simple_history.history["val_accuracy"])
    histories["Simple LSTM"] = simple_history
    
    # 2. 增强LSTM + 3D特征
    print("=" * 60)
    print("训练增强LSTM + 3D特征模型...")
    print("=" * 60)
    
    enhanced_lstm = Trajectory3DProcessor(input_shape)
    enhanced_history = enhanced_lstm.train(X_train, y_train, X_val, y_val, 
                                          epochs=50, batch_size=32, verbose=1, 
                                          use_spatial_features=True)
    results["Enhanced LSTM + 3D Features"] = max(enhanced_history.history["val_accuracy"])
    histories["Enhanced LSTM + 3D Features"] = enhanced_history
    
    # 3. 传统ML + 3D特征
    print("=" * 60)
    print("训练传统ML模型 + 3D特征...")
    print("=" * 60)
    
    processor = Trajectory3DProcessor(input_shape)
    spatial_features, _, _ = processor.preprocess_trajectories(X_train)
    spatial_features_val, _, _ = processor.preprocess_trajectories(X_val)
    
    # 随机森林
    print("训练随机森林...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(spatial_features, y_train)
    rf_pred = rf.predict(spatial_features_val)
    results["Random Forest + 3D Features"] = accuracy_score(y_val, rf_pred)
    
    # SVM
    print("训练SVM...")
    svm = SVC(kernel="rbf", random_state=42)
    svm.fit(spatial_features, y_train)
    svm_pred = svm.predict(spatial_features_val)
    results["SVM + 3D Features"] = accuracy_score(y_val, svm_pred)
    
    # 4. 可视化比较结果
    print("=" * 60)
    print("生成可视化结果...")
    print("=" * 60)
    
    # 架构性能比较
    plt.figure(figsize=(12, 8))
    models = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(models, accuracies, color=["skyblue", "lightgreen", "salmon", "orange"])
    plt.title("3D轨迹处理架构性能比较", fontsize=16, fontweight="bold")
    plt.ylabel("验证准确率", fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/architecture_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 训练过程比较
    plt.figure(figsize=(15, 5))
    
    # 准确率曲线
    plt.subplot(1, 3, 1)
    for name, history in histories.items():
        plt.plot(history.history["accuracy"], label=f"{name} (Train)", linestyle="-")
        plt.plot(history.history["val_accuracy"], label=f"{name} (Val)", linestyle="--")
    plt.title("训练准确率比较")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 损失曲线
    plt.subplot(1, 3, 2)
    for name, history in histories.items():
        plt.plot(history.history["loss"], label=f"{name} (Train)", linestyle="-")
        plt.plot(history.history["val_loss"], label=f"{name} (Val)", linestyle="--")
    plt.title("训练损失比较")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最终性能对比
    plt.subplot(1, 3, 3)
    final_accuracies = [max(history.history["val_accuracy"]) for history in histories.values()]
    model_names = list(histories.keys())
    bars = plt.bar(model_names, final_accuracies, color=["skyblue", "lightgreen"])
    plt.title("最终验证准确率")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    for bar, acc in zip(bars, final_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 混淆矩阵比较
    plt.figure(figsize=(15, 5))
    
    # 简单LSTM混淆矩阵
    plt.subplot(1, 3, 1)
    simple_pred = simple_lstm.predict(X_val, use_spatial_features=False)
    simple_pred_labels = np.argmax(simple_pred, axis=1)
    cm_simple = confusion_matrix(y_val, simple_pred_labels)
    sns.heatmap(cm_simple, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Patient1", "Patient2"],
                yticklabels=["Normal", "Patient1", "Patient2"])
    plt.title("Simple LSTM")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # 增强LSTM混淆矩阵
    plt.subplot(1, 3, 2)
    enhanced_pred = enhanced_lstm.predict(X_val, use_spatial_features=True)
    enhanced_pred_labels = np.argmax(enhanced_pred, axis=1)
    cm_enhanced = confusion_matrix(y_val, enhanced_pred_labels)
    sns.heatmap(cm_enhanced, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Normal", "Patient1", "Patient2"],
                yticklabels=["Normal", "Patient1", "Patient2"])
    plt.title("Enhanced LSTM + 3D Features")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # 随机森林混淆矩阵
    plt.subplot(1, 3, 3)
    cm_rf = confusion_matrix(y_val, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Oranges",
                xticklabels=["Normal", "Patient1", "Patient2"],
                yticklabels=["Normal", "Patient1", "Patient2"])
    plt.title("Random Forest + 3D Features")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 保存详细结果
    print("=" * 60)
    print("保存详细结果...")
    print("=" * 60)
    
    # 保存结果到文件
    with open(f"{save_path}/comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("3D轨迹处理架构性能比较结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 架构性能排名:\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (model, acc) in enumerate(sorted_results, 1):
            f.write(f"{i}. {model}: {acc:.4f}\n")
        
        f.write(f"\n2. 最佳模型: {sorted_results[0][0]}\n")
        f.write(f"   准确率: {sorted_results[0][1]:.4f}\n")
        
        f.write(f"\n3. 详细性能报告:\n")
        for model, acc in results.items():
            f.write(f"   {model}: {acc:.4f}\n")
    
    # 保存分类报告
    with open(f"{save_path}/classification_reports.txt", "w", encoding="utf-8") as f:
        f.write("分类报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 简单LSTM报告
        f.write("Simple LSTM:\n")
        f.write(classification_report(y_val, simple_pred_labels, 
                                    target_names=["Normal", "Patient1", "Patient2"]))
        f.write("\n" + "-" * 50 + "\n\n")
        
        # 增强LSTM报告
        f.write("Enhanced LSTM + 3D Features:\n")
        f.write(classification_report(y_val, enhanced_pred_labels,
                                    target_names=["Normal", "Patient1", "Patient2"]))
        f.write("\n" + "-" * 50 + "\n\n")
        
        # 随机森林报告
        f.write("Random Forest + 3D Features:\n")
        f.write(classification_report(y_val, rf_pred,
                                    target_names=["Normal", "Patient1", "Patient2"]))
    
    print(f"所有结果已保存到 {save_path} 文件夹")
    print(f"包含文件:")
    print(f"  - architecture_comparison.png: 架构性能比较图")
    print(f"  - training_comparison.png: 训练过程比较图")
    print(f"  - confusion_matrices.png: 混淆矩阵比较图")
    print(f"  - comparison_results.txt: 详细性能结果")
    print(f"  - classification_reports.txt: 分类报告")
    
    return results, histories

if __name__ == "__main__":
    # 示例使用
    input_shape = (200, 3)  # 时间步长200，3D坐标
    processor = Trajectory3DProcessor(input_shape)
    
    # 构建简单模型
    simple_model = processor.build_simple_lstm_model()
    simple_model.summary()
    
    print("\n" + "="*50)
    
    # 构建增强模型
    enhanced_model = processor.build_enhanced_lstm_model()
    enhanced_model.summary()
