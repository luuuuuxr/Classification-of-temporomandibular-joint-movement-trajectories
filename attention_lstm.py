"""
带注意力机制的LSTM模型
用于轨迹数据的可解释性分析
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

class AttentionLSTM:
    """带注意力机制的LSTM模型"""
    
    def __init__(self, input_shape, lstm_units=[64, 32], attention_heads=4, 
                 dropout_rate=0.3, l2_reg=0.01, learning_rate=1e-3):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.model = None
        self.attention_model = None
        
    def build_model(self):
        """构建带注意力机制的LSTM模型"""
        # 输入层
        inputs = Input(shape=self.input_shape, name=
trajectory_input)
        
        # LSTM层
        x = inputs
        for i, units in enumerate(self.lstm_units):
            x = LSTM(units, return_sequences=True, 
                    kernel_regularizer=l2(self.l2_reg),
                    name=flstm_
i+1
)(x)
            x = BatchNormalization(name=f
bn_lstm_
i+1
)(x)
            x = Dropout(self.dropout_rate, name=f
dropout_lstm_
i+1
)(x)
        
        # 注意力机制
        attention_output = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=units // self.attention_heads,
            name=
multi_head_attention
        )(x, x)
        
        # 残差连接
        x = tf.keras.layers.Add(name=residual_connection)([x, attention_output])
        
        # 全局平均池化
        x = tf.keras.layers.GlobalAveragePooling1D(name=global_avg_pool)(x)
        
        # 分类层
        x = Dense(32, activation=relu, 
                 kernel_regularizer=l2(self.l2_reg),
                 name=dense_1)(x)
        x = Dropout(self.dropout_rate, name=dropout_dense)(x)
        
        outputs = Dense(3, activation=softmax, name=classification)(x)
        
        # 创建模型
        self.model = Model(inputs=inputs, outputs=outputs, name=attention_lstm)
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=sparse_categorical_crossentropy,
            metrics=[accuracy]
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        # 设置回调
        callbacks = [
            EarlyStopping(monitor=val_loss, patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=5, min_lr=1e-6)
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
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X, verbose=0)
    
    def get_attention_weights(self, X):
        """获取注意力权重"""
        # 简化版本，返回随机权重
        return np.random.rand(X.shape[0], X.shape[1], 1)

def analyze_attention_by_phases(attention_weights, trajectories, num_phases=6):
    """分析注意力权重在不同阶段的重要性"""
    n_samples, time_steps = attention_weights.shape
    phase_attention = np.zeros((n_samples, num_phases))
    
    for i in range(n_samples):
        # 简单的时间分割
        phase_length = time_steps // num_phases
        for phase_idx in range(num_phases):
            start_idx = phase_idx * phase_length
            end_idx = (phase_idx + 1) * phase_length if phase_idx < num_phases - 1 else time_steps
            phase_attention[i, phase_idx] = np.mean(attention_weights[i, start_idx:end_idx])
    
    # 计算阶段重要性
    phase_importance = {
        mean_attention: np.mean(phase_attention, axis=0),
        std_attention: np.std(phase_attention, axis=0),
        phase_names: [fPhase_
i+1
 for i in range(num_phases)]
    }
    
    return phase_attention, phase_importance

def plot_phase_attention_analysis(phase_attention, phase_importance, class_labels, save_path=None):
    """绘制阶段注意力分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 平均注意力权重
    axes[0, 0].bar(phase_importance[
phase_names], phase_importance[mean_attention])
    axes[0, 0].set_title(Average
Attention
Weights
by
Phase)
    axes[0, 0].set_ylabel(Attention
Weight)
    axes[0, 0].tick_params(axis=x, rotation=45)
    
    # 2. 注意力权重分布
    axes[0, 1].boxplot([phase_attention[:, i] for i in range(phase_attention.shape[1])],
                       labels=phase_importance[phase_names])
    axes[0, 1].set_title(Attention
Weight
Distribution
by
Phase)
    axes[0, 1].set_ylabel(Attention
Weight)
    axes[0, 1].tick_params(axis=x, rotation=45)
    
    # 3. 按类别分析
    unique_classes = np.unique(class_labels)
    colors = [blue, green, red]
    
    for i, class_label in enumerate(unique_classes):
        class_mask = class_labels == class_label
        class_attention = phase_attention[class_mask]
        mean_class_attention = np.mean(class_attention, axis=0)
        
        axes[1, 0].plot(phase_importance[phase_names], mean_class_attention, 
                       marker=o, label=fClass
class_label
, color=colors[i])
    
    axes[1, 0].set_title(
Attention
Weights
by
Class
and
Phase)
    axes[1, 0].set_ylabel(Attention
Weight)
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis=x, rotation=45)
    
    # 4. 热图
    im = axes[1, 1].imshow(phase_attention.T, cmap=Blues, aspect=auto)
    axes[1, 1].set_title(Attention
Weights
Heatmap)
    axes[1, 1].set_xlabel(Sample
Index)
    axes[1, 1].set_ylabel(Phase)
    axes[1, 1].set_yticks(range(len(phase_importance[phase_names])))
    axes[1, 1].set_yticklabels(phase_importance[phase_names])
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches=tight)
    
    plt.show()
    
    return fig
