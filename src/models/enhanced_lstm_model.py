import numpy as np
import tensorflow as tf
from keras import Model, Input
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, MultiHeadAttention, Add, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, LayerNormalization, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

class EnhancedLSTMFeatureExtractor:
    def __init__(self, lstm_units=[128, 64], attention_heads=8, key_dim=64, dense_units=128, dropout_rate=0.3, l2_reg=0.01):
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.key_dim = key_dim
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        self.scaler = StandardScaler()
        
    def build_enhanced_lstm(self, input_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(64, 3, padding="same", activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        lstm1 = Bidirectional(LSTM(self.lstm_units[0], return_sequences=True, kernel_regularizer=l2(self.l2_reg)), merge_mode="concat")(x)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        attention_out = MultiHeadAttention(num_heads=self.attention_heads, key_dim=self.key_dim, dropout=self.dropout_rate)(lstm1, lstm1)
        residual1 = Add()([lstm1, attention_out])
        residual1 = LayerNormalization()(residual1)
        lstm2 = Bidirectional(LSTM(self.lstm_units[1], return_sequences=True, kernel_regularizer=l2(self.l2_reg)), merge_mode="concat")(residual1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(self.dropout_rate)(lstm2)
        self_attention = MultiHeadAttention(num_heads=self.attention_heads // 2, key_dim=self.key_dim, dropout=self.dropout_rate)(lstm2, lstm2)
        residual2 = Add()([lstm2, self_attention])
        residual2 = LayerNormalization()(residual2)
        avg_pool = GlobalAveragePooling1D()(residual2)
        max_pool = GlobalMaxPooling1D()(residual2)
        combined = Concatenate()([avg_pool, max_pool])
        dense1 = Dense(self.dense_units, activation="relu", kernel_regularizer=l2(self.l2_reg))(combined)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.dropout_rate)(dense1)
        dense2 = Dense(self.dense_units // 2, activation="relu", kernel_regularizer=l2(self.l2_reg))(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(self.dropout_rate * 0.5)(dense2)
        output = Dense(64, activation="linear", kernel_regularizer=l2(self.l2_reg))(dense2)
        model = Model(inputs, output)
        return model

    def build_classification_model(self, input_shape, num_classes=3):
        inputs = Input(shape=input_shape)
        feature_extractor = self.build_enhanced_lstm(input_shape)
        features = feature_extractor(inputs)
        classification_head = Dense(32, activation="relu")(features)
        classification_head = Dropout(0.2)(classification_head)
        classification_output = Dense(num_classes, activation="softmax")(classification_head)
        model = Model(inputs, classification_output)
        return model
    
    def train_feature_extractor(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, save_path="enhanced_lstm_model.h5"):
        self.model = self.build_classification_model(X_train.shape[1:])
        self.model.compile(optimizer=Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        callbacks = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True), ReduceLROnPlateau(monitor="val_accuracy", factor=0.7, patience=5, min_lr=1e-6), ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True, verbose=1)]
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
        feature_model = self.build_enhanced_lstm(X_train.shape[1:])
        for i, layer in enumerate(feature_model.layers[1:-1]):
            if i < len(self.model.layers) - 1:
                try:
                    layer.set_weights(self.model.layers[i+1].get_weights())
                except:
                    pass
        self.model = feature_model
        return history
    
    def extract_features(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_feature_extractor")
        features = self.model.predict(X, batch_size=128, verbose=0)
        return features
    
    def extract_features_with_scaling(self, X_train, X_test):
        train_features = self.extract_features(X_train)
        test_features = self.extract_features(X_test)
        train_features_scaled = self.scaler.fit_transform(train_features)
        test_features_scaled = self.scaler.transform(test_features)
        return train_features_scaled, test_features_scaled

def create_enhanced_lstm_model(input_shape, lstm_units=[128, 64], attention_heads=8, dense_units=128, dropout_rate=0.3):
    extractor = EnhancedLSTMFeatureExtractor(lstm_units=lstm_units, attention_heads=attention_heads, dense_units=dense_units, dropout_rate=dropout_rate)
    return extractor.build_enhanced_lstm(input_shape)
