
# 读取文件
with open('main_improved.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到要替换的行
start_idx = -1
for i, line in enumerate(lines):
    if 'X_train_features, X_test_features = extract_lstm_features' in line:
        start_idx = i
        break

# 找到结束行
end_idx = -1
for i in range(start_idx, len(lines)):
    if ')' in lines[i] and 'use_advanced_attention=True' in lines[i-1]:
        end_idx = i
        break

if start_idx >= 0 and end_idx >= 0:
    # 替换代码
    new_code = [
        '    # ============ 特征提取 ============\n',
        '    if USE_ENHANCED_LSTM:\n',
        '        print(\
🚀
使用增强LSTM进行特征提取...\)\n',
        '        # 使用增强LSTM特征提取\n',
        '        extractor = EnhancedLSTMFeatureExtractor(\n',
        '            lstm_units=[128, 64],\n',
        '            attention_heads=8,\n',
        '            dense_units=128,\n',
        '            dropout_rate=0.3\n',
        '        )\n',
        '        \n',
        '        # 训练特征提取器\n',
        '        print(\📚
训练增强LSTM特征提取器...\)\n',
        '        history = extractor.train_feature_extractor(\n',
        '            X_train, y_train, X_val, y_val,\n',
        '            epochs=100,\n',
        '            batch_size=64,\n',
        '            save_path=\enhanced_lstm_model.h5\\n',
        '        )\n',
        '        \n',
        '        # 提取特征\n',
        '        print(\🔍
提取训练集和测试集特征...\)\n',
        '        X_train_features, X_test_features = extractor.extract_features_with_scaling(X_train, X_test)\n',
        '        \n',
        '        print(f\✅
增强LSTM特征提取完成!\)\n',
        '        print(f\训练集特征形状:
X_train_features.shape
\)\n',
        '        print(f\测试集特征形状:
X_test_features.shape
\)\n',
        '    else:\n',
        '        print(\🔍
使用原始LSTM进行特征提取...\)\n',
        '        X_train_features, X_test_features = extract_lstm_features(X_train, y_train, X_val, y_val, X_test_scaler,\n',
        '                                                                  input_shape=None,\n',
        '                                                                  save_path=\best_model.h5\,\n',
        '                                                                  dense_units=32,  # ✅ 输出维度\n',
        '                                                                  use_attention=True,  # ✅ 是否使用 Attention\n',
        '                                                                  loss_type=\mse\,  # ✅ 损失函数类型\n',
        '                                                                  return_scaler=False,  # ✅ 是否返回 scaler\n',
        '                                                                  use_advanced_attention=True  # ✅ 启用增强注意力机制\n',
        '                                                                  )\n',
        '    \n'
    ]
    
    # 替换代码
    new_lines = lines[:start_idx] + new_code + lines[end_idx+1:]
    
    # 写回文件
    with open('main_improved.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print('✅ 已修改特征提取部分，现在会使用USE_ENHANCED_LSTM变量')
else:
    print('❌ 未找到要替换的代码段')

