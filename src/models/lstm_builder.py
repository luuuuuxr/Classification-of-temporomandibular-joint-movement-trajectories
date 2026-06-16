# -*- coding: utf-8 -*-
"""
LSTM 模型构建模块：
  residual_block / build_lstm_feature_model_final / extract_lstm_features
"""

import tensorflow as tf
from keras import Model, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import (LSTM, Dense, Dropout, BatchNormalization, Attention,
                           Bidirectional, Concatenate, GlobalAveragePooling1D)
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler

from src.models.advanced_attention import AdvancedAttentionMechanisms


def residual_block(x, units=64, dropout_rate=0.25, l2_reg=0.001):
    shortcut = x
    x = Bidirectional(LSTM(units // 2, return_sequences=True,
                           kernel_regularizer=l2(l2_reg)))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = Dense(x.shape[-1], activation=None)(shortcut)
    return tf.keras.layers.add([shortcut, x])


def build_lstm_feature_model_final(input_shape,
                                   lstm_units=64,
                                   dense_units=32,
                                   use_attention=True,
                                   use_advanced_attention=False):
    inputs = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=True,
             kernel_regularizer=l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = residual_block(x, lstm_units * 2, 0.1, 0.001)
    x = residual_block(x, lstm_units, 0.2, 0.001)

    if use_advanced_attention:
        attention_mechanisms = AdvancedAttentionMechanisms()
        x = attention_mechanisms.multi_scale_attention(x, scales=[1, 2, 4])
        x, _ = attention_mechanisms.temporal_attention_layer(x)
        x, _ = attention_mechanisms.spatial_attention_layer(x)
        x = attention_mechanisms.self_attention_layer(
            x, num_heads=8, key_dim=lstm_units // 8)
    elif use_attention:
        attn = Attention()([x, x])
        x = Concatenate()([x, attn])

    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(0.01))(x)

    num_classes = 3
    classification_output = Dense(num_classes, activation='softmax',
                                  name='classification')(x)
    model = Model(inputs, classification_output)
    feature_model = Model(inputs, x)
    return model, feature_model


def extract_lstm_features(X_train, y_train, X_val, y_val, X_test,
                          input_shape=None,
                          save_path="best_model.h5",
                          dense_units=32,
                          use_attention=True,
                          use_advanced_attention=False,
                          loss_type='sparse_categorical_crossentropy',
                          return_scaler=False):
    if input_shape is None:
        input_shape = X_train.shape[1:]

    model, feature_model = build_lstm_feature_model_final(
        input_shape=input_shape,
        lstm_units=64,
        dense_units=dense_units,
        use_attention=use_attention,
        use_advanced_attention=use_advanced_attention
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=loss_type,
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=7,
                                   restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.7,
                                     patience=5, min_lr=5e-5)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss',
                                 save_best_only=True, verbose=1)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, lr_scheduler, checkpoint],
        verbose=2
    )

    X_train_features = feature_model.predict(X_train, batch_size=128)
    X_val_features = feature_model.predict(X_val, batch_size=128)
    X_test_features = feature_model.predict(X_test, batch_size=128)

    scaler = StandardScaler()
    X_train_features = scaler.fit_transform(X_train_features)
    X_val_features = scaler.transform(X_val_features)
    X_test_features = scaler.transform(X_test_features)

    if return_scaler:
        return X_train_features, X_val_features, X_test_features, scaler
    return X_train_features, X_val_features, X_test_features
