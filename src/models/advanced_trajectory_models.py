"""
先进时序模型 - 专为3D轨迹数据设计
包含适合3D运动轨迹分类的先进深度学习模型
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Conv1D, MaxPooling1D, 
    Dense, Dropout, BatchNormalization, GlobalAveragePooling1D,
    GlobalMaxPooling1D, Concatenate, MultiHeadAttention,
    LayerNormalization, Add, Reshape, Permute
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class TrajectoryTransformer:
    """基于Transformer的3D轨迹分类器"""
    
    def __init__(self, input_shape, num_classes=3, d_model=128, num_heads=8, 
                 num_layers=4, dff=512, dropout_rate=0.1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self):
        """构建Transformer模型"""
        inputs = Input(shape=self.input_shape, name='trajectory_input')
        
        # 位置编码
        x = self._positional_encoding(inputs)
        
        # Transformer编码器层
        for i in range(self.num_layers):
            x = self._transformer_encoder_layer(x, f'encoder_{i}')
        
        # 全局池化
        x = GlobalAveragePooling1D()(x)
        
        # 分类头
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def _positional_encoding(self, x):
        """位置编码"""
        seq_len = self.input_shape[0]
        d_model = self.d_model
        
        # 创建位置编码矩阵
        pos_encoding = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        
        pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
        
        # 将输入投影到d_model维度
        x = Dense(self.d_model)(x)
        x = x + pos_encoding
        
        return x
    
    def _transformer_encoder_layer(self, x, name):
        """Transformer编码器层"""
        # 多头自注意力
        attn_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            name=f'{name}_attention'
        )(x, x)
        attn_output = Dropout(self.dropout_rate)(attn_output)
        x = LayerNormalization()(x + attn_output)
        
        # 前馈网络
        ffn = Dense(self.dff, activation='relu')(x)
        ffn = Dropout(self.dropout_rate)(ffn)
        ffn = Dense(self.d_model)(ffn)
        ffn = Dropout(self.dropout_rate)(ffn)
        x = LayerNormalization()(x + ffn)
        
        return x
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=5, min_lr=1e-6),
            ModelCheckpoint('transformer_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        report = classification_report(y_test, y_pred_classes)
        cm = confusion_matrix(y_test, y_pred_classes)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred_classes,
            'probabilities': y_pred
        }


class CNNLSTMModel:
    """CNN-LSTM混合模型 - 适合3D轨迹数据"""
    
    def __init__(self, input_shape, num_classes=3, cnn_filters=64, 
                 lstm_units=128, dropout_rate=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self):
        """构建CNN-LSTM模型"""
        inputs = Input(shape=self.input_shape, name='trajectory_input')
        
        # CNN特征提取
        x = Conv1D(filters=self.cnn_filters, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(filters=self.cnn_filters*2, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(filters=self.cnn_filters*4, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # LSTM时序建模
        x = LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate)(x)
        x = LSTM(self.lstm_units//2, dropout=self.dropout_rate)(x)
        
        # 分类头
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=5, min_lr=1e-6),
            ModelCheckpoint('cnn_lstm_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        report = classification_report(y_test, y_pred_classes)
        cm = confusion_matrix(y_test, y_pred_classes)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred_classes,
            'probabilities': y_pred
        }


class BiLSTMAttentionModel:
    """双向LSTM + 注意力机制模型"""
    
    def __init__(self, input_shape, num_classes=3, lstm_units=128, 
                 attention_heads=8, dropout_rate=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self):
        """构建双向LSTM + 注意力模型"""
        inputs = Input(shape=self.input_shape, name='trajectory_input')
        
        # 双向LSTM
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate))(inputs)
        x = BatchNormalization()(x)
        
        x = Bidirectional(LSTM(self.lstm_units//2, return_sequences=True, dropout=self.dropout_rate))(x)
        x = BatchNormalization()(x)
        
        # 多头注意力
        x = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.lstm_units//self.attention_heads
        )(x, x)
        
        # 全局池化
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = Concatenate()([avg_pool, max_pool])
        
        # 分类头
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=5, min_lr=1e-6),
            ModelCheckpoint('bilstm_attention_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        report = classification_report(y_test, y_pred_classes)
        cm = confusion_matrix(y_test, y_pred_classes)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred_classes,
            'probabilities': y_pred
        }


class TrajectoryModelTrainer:
    """轨迹模型训练器"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_all_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """训练所有先进时序模型"""
        print("🚀 开始训练先进时序模型...")
        
        # 1. Transformer模型
        print("\n🤖 训练Transformer模型...")
        try:
            transformer = TrajectoryTransformer(X_train.shape[1:])
            transformer.train(X_train, y_train, X_val, y_val)
            transformer_result = transformer.evaluate(X_test, y_test)
            
            self.models['Transformer'] = transformer
            self.results['Transformer'] = {
                'train_acc': transformer_result['accuracy'],
                'test_acc': transformer_result['accuracy'],
                'confusion_matrix': transformer_result['confusion_matrix']
            }
            print(f"Transformer - 测试准确率: {transformer_result['accuracy']:.4f}")
        except Exception as e:
            print(f"Transformer训练失败: {e}")
        
        # 2. CNN-LSTM模型
        print("\n🔄 训练CNN-LSTM模型...")
        try:
            cnn_lstm = CNNLSTMModel(X_train.shape[1:])
            cnn_lstm.train(X_train, y_train, X_val, y_val)
            cnn_lstm_result = cnn_lstm.evaluate(X_test, y_test)
            
            self.models['CNN-LSTM'] = cnn_lstm
            self.results['CNN-LSTM'] = {
                'train_acc': cnn_lstm_result['accuracy'],
                'test_acc': cnn_lstm_result['accuracy'],
                'confusion_matrix': cnn_lstm_result['confusion_matrix']
            }
            print(f"CNN-LSTM - 测试准确率: {cnn_lstm_result['accuracy']:.4f}")
        except Exception as e:
            print(f"CNN-LSTM训练失败: {e}")
        
        # 3. 双向LSTM + 注意力模型
        print("\n↔️ 训练双向LSTM+注意力模型...")
        try:
            bilstm_attn = BiLSTMAttentionModel(X_train.shape[1:])
            bilstm_attn.train(X_train, y_train, X_val, y_val)
            bilstm_result = bilstm_attn.evaluate(X_test, y_test)
            
            self.models['BiLSTM-Attention'] = bilstm_attn
            self.results['BiLSTM-Attention'] = {
                'train_acc': bilstm_result['accuracy'],
                'test_acc': bilstm_result['accuracy'],
                'confusion_matrix': bilstm_result['confusion_matrix']
            }
            print(f"BiLSTM-Attention - 测试准确率: {bilstm_result['accuracy']:.4f}")
        except Exception as e:
            print(f"BiLSTM-Attention训练失败: {e}")
        
        return self.results
    
    def get_best_model(self):
        """获取最佳模型"""
        if not self.results:
            return None, 0
        
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_acc'])
        best_accuracy = self.results[best_model_name]['test_acc']
        
        return best_model_name, best_accuracy


def test_advanced_trajectory_models(X_train, y_train, X_test, y_test):
    """测试先进时序模型"""
    # 分割训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 创建训练器
    trainer = TrajectoryModelTrainer()
    
    # 训练所有模型
    results = trainer.train_all_models(
        X_train_split, y_train_split, X_val_split, y_val_split, X_test, y_test
    )
    
    # 获取最佳模型
    best_model_name, best_accuracy = trainer.get_best_model()
    
    return trainer, results, (best_model_name, best_accuracy)
