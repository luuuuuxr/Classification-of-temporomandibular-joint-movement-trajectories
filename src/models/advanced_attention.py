"""
先进的注意力机制模块
包含多种注意力机制用于轨迹数据分类
"""
import tensorflow as tf
from keras import Model, Input
from keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, MultiHeadAttention,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D,
    Add, LayerNormalization, Concatenate, Reshape
)

# 修复 AdamW 导入问题
try:
    from keras.optimizers import AdamW
except ImportError:
    try:
        from keras.optimizers.legacy import AdamW
    except ImportError:
        # 如果都找不到，使用 Adam 作为替代
        from keras.optimizers import Adam as AdamW
from keras.regularizers import l2


class AdvancedAttentionMechanisms:
    """先进的注意力机制集合"""

    @staticmethod
    def self_attention_layer(x, num_heads=8, key_dim=64, name="self_attention"):
        """自注意力机制"""
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            name=name
        )(x, x)

        # 残差连接和层归一化
        attention_output = Add(name=f"{name}_residual")([x, attention_output])
        attention_output = LayerNormalization(name=f"{name}_norm")(attention_output)

        return attention_output

    @staticmethod
    def cross_attention_layer(query, key_value, num_heads=8, key_dim=64, name="cross_attention"):
        """交叉注意力机制"""
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            name=name
        )(query, key_value)

        # 残差连接和层归一化
        attention_output = Add(name=f"{name}_residual")([query, attention_output])
        attention_output = LayerNormalization(name=f"{name}_norm")(attention_output)

        return attention_output

    @staticmethod
    def temporal_attention_layer(x, name="temporal_attention"):
        """时间注意力机制"""
        # 计算时间步的注意力权重
        attention_weights = Dense(1, activation="softmax", name=f"{name}_weights")(x)

        # 应用注意力权重
        attended_output = tf.keras.layers.Multiply(name=f"{name}_multiply")([x, attention_weights])

        return attended_output, attention_weights

    @staticmethod
    def spatial_attention_layer(x, name="spatial_attention"):
        """空间注意力机制（针对特征维度）"""
        # 获取特征维度
        feature_dim = x.shape[-1]

        # 对每个特征维度计算注意力权重
        attention_weights = Dense(feature_dim, activation="softmax", name=f"{name}_weights")(x)

        # 应用注意力权重
        attended_output = tf.keras.layers.Multiply(name=f"{name}_multiply")([x, attention_weights])

        return attended_output, attention_weights

    @staticmethod
    def multi_scale_attention(x, scales=[1, 2, 4], name="multi_scale_attention"):
        """多尺度注意力机制"""
        attention_outputs = []

        for scale in scales:
            # 不同尺度的卷积
            conv = Conv1D(
                filters=x.shape[-1] // len(scales),
                kernel_size=scale * 2 + 1,
                padding="same",
                activation="relu",
                name=f"{name}_conv_{scale}"
            )(x)

            # 注意力权重
            attention_weights = Dense(1, activation="softmax", name=f"{name}_weights_{scale}")(conv)

            # 应用注意力
            attended = tf.keras.layers.Multiply(name=f"{name}_multiply_{scale}")([conv, attention_weights])
            attention_outputs.append(attended)

        # 拼接不同尺度的结果
        multi_scale_output = Concatenate(name=f"{name}_concat")(attention_outputs)

        return multi_scale_output

    @staticmethod
    def channel_attention_layer(x, reduction_ratio=16, name="channel_attention"):
        """通道注意力机制"""
        # 全局平均池化和最大池化
        avg_pool = GlobalAveragePooling1D(name=f"{name}_avg_pool")(x)
        max_pool = GlobalMaxPooling1D(name=f"{name}_max_pool")(x)

        # 共享的MLP
        def shared_mlp(inputs, name_prefix):
            x = Dense(inputs.shape[-1] // reduction_ratio, activation="relu",
                      name=f"{name_prefix}_dense1")(inputs)
            x = Dense(inputs.shape[-1], activation="sigmoid",
                      name=f"{name_prefix}_dense2")(x)
            return x

        avg_out = shared_mlp(avg_pool, f"{name}_avg")
        max_out = shared_mlp(max_pool, f"{name}_max")

        # 合并
        channel_attention = tf.keras.layers.Add(name=f"{name}_add")([avg_out, max_out])
        channel_attention = tf.keras.layers.Activation("sigmoid", name=f"{name}_sigmoid")(channel_attention)

        # 重塑为正确的形状
        channel_attention = Reshape((1, -1), name=f"{name}_reshape")(channel_attention)

        # 应用通道注意力
        attended_output = tf.keras.layers.Multiply(name=f"{name}_multiply")([x, channel_attention])

        return attended_output, channel_attention

    @staticmethod
    def positional_attention_layer(x, name="positional_attention"):
        """位置注意力机制"""
        # 计算位置编码
        seq_len = tf.shape(x)[1]
        d_model = x.shape[-1]

        # 创建位置编码
        pos_encoding = tf.range(seq_len, dtype=tf.float32)
        pos_encoding = tf.expand_dims(pos_encoding, 1)
        pos_encoding = tf.tile(pos_encoding, [1, d_model])

        # 位置编码投影
        pos_proj = Dense(d_model, activation="tanh", name=f"{name}_pos_proj")(pos_encoding)

        # 计算位置注意力权重
        attention_weights = Dense(1, activation="softmax", name=f"{name}_weights")(pos_proj)

        # 应用位置注意力
        attended_output = tf.keras.layers.Multiply(name=f"{name}_multiply")([x, attention_weights])

        return attended_output, attention_weights


class EnhancedAttentionLSTM:
    """增强的注意力LSTM模型"""

    def __init__(self, input_shape, num_classes=3, lstm_units=[128, 64, 32],
                 attention_heads=8, dropout_rate=0.3, l2_reg=0.01):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.attention_mechanisms = AdvancedAttentionMechanisms()
        self.model = None

    def build_enhanced_model(self, name="enhanced_attention_lstm"):
        """构建增强的注意力LSTM模型"""
        inputs = Input(shape=self.input_shape, name="trajectory_input")

        # CNN特征提取
        x = self._build_cnn_backbone(inputs)

        # 双向LSTM
        x = tf.keras.layers.Bidirectional(
            LSTM(self.lstm_units[0], return_sequences=True),
            name="bilstm_1"
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        # 多尺度注意力
        x = self.attention_mechanisms.multi_scale_attention(x)

        # 时间注意力
        x, time_weights = self.attention_mechanisms.temporal_attention_layer(x)

        # 空间注意力
        x, spatial_weights = self.attention_mechanisms.spatial_attention_layer(x)

        # 通道注意力
        x, channel_weights = self.attention_mechanisms.channel_attention_layer(x)

        # 自注意力
        x = self.attention_mechanisms.self_attention_layer(
            x, num_heads=self.attention_heads
        )

        # 多尺度池化
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = Concatenate()([avg_pool, max_pool])

        # 分类层
        x = Dense(128, activation="relu", kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation="relu", kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs, outputs, name=name)
        return model

    def build_residual_attention_model(self, name="residual_attention_lstm"):
        """构建残差注意力LSTM模型"""
        inputs = Input(shape=self.input_shape, name="trajectory_input")

        x = inputs
        lstm_outputs = []

        # 多层残差LSTM
        for i, units in enumerate(self.lstm_units):
            # LSTM层
            lstm_out = LSTM(units, return_sequences=True,
                            kernel_regularizer=l2(self.l2_reg))(x)
            lstm_out = BatchNormalization()(lstm_out)
            lstm_out = Dropout(self.dropout_rate)(lstm_out)

            # 残差连接
            if i > 0 and x.shape[-1] == units:
                lstm_out = Add()([x, lstm_out])

            # 注意力机制
            if i == len(self.lstm_units) - 1:  # 最后一层添加注意力
                lstm_out = self.attention_mechanisms.self_attention_layer(
                    lstm_out, num_heads=self.attention_heads
                )

            x = lstm_out
            lstm_outputs.append(lstm_out)

        # 特征融合
        if len(lstm_outputs) > 1:
            x = Concatenate()(lstm_outputs[-2:])  # 融合最后两层

        # 全局池化
        x = GlobalAveragePooling1D()(x)

        # 分类层
        x = Dense(64, activation="relu", kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs, outputs, name=name)
        return model

    def _build_cnn_backbone(self, inputs):
        """构建CNN骨干网络"""
        x = inputs

        # 多尺度卷积
        conv1 = Conv1D(64, 3, padding="same", activation="relu")(x)
        conv2 = Conv1D(64, 5, padding="same", activation="relu")(x)
        conv3 = Conv1D(64, 7, padding="same", activation="relu")(x)

        # 拼接多尺度特征
        x = Concatenate()([conv1, conv2, conv3])
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        # 更深的卷积层
        x = Conv1D(128, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        return x

    def compile_model(self, model, learning_rate=1e-3):
        """编译模型"""
        model.compile(
            optimizer=AdamW(learning_rate=learning_rate, weight_decay=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model


# 使用示例
def create_attention_models(input_shape, num_classes=3):
    """创建各种注意力模型"""
    enhanced_lstm = EnhancedAttentionLSTM(input_shape, num_classes)

    models = {
        "enhanced_attention": enhanced_lstm.build_enhanced_model(),
        "residual_attention": enhanced_lstm.build_residual_attention_model()
    }

    # 编译模型
    for name, model in models.items():
        enhanced_lstm.compile_model(model)

    return models


if __name__ == "__main__":
    # 测试注意力机制
    input_shape = (200, 3)
    models = create_attention_models(input_shape)

    for name, model in models.items():
        print(f"{name} 模型结构:")
        model.summary()
        print("=" * 50)
