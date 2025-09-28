# 修复脚本
import re

# 读取文件内容
with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# 找到需要替换的代码段
old_pattern = r"    else:\s*# Keras 模型（如 BoNet）\s*y_train_prob = model\.predict\(X_train_features\)\s*y_test_prob = model\.predict\(X_test_features\)\s*y_train_pred = np\.argmax\(y_train_prob, axis=1\)\s*y_test_pred = np\.argmax\(y_test_prob, axis=1\)"

new_code = """    else:
        # 其他模型（如 BoNet, PrototypicalNetwork 等）
        y_train_pred = model.predict(X_train_features)
        y_test_pred = model.predict(X_test_features)
        
        # 检查预测结果是否为概率分布（2D数组）
        if isinstance(y_train_pred, np.ndarray) and y_train_pred.ndim > 1:
            # 如果是概率分布，提取类别预测
            y_train_pred = np.argmax(y_train_pred, axis=1)
            y_test_pred = np.argmax(y_test_pred, axis=1)
            y_train_prob = y_train_pred  # 使用原始概率分布
            y_test_prob = y_test_pred
        else:
            # 如果已经是类别索引（1D数组），无法计算AUC
            y_train_prob = None
            y_test_prob = None"""

# 替换代码
if re.search(old_pattern, content, re.MULTILINE | re.DOTALL):
    content = re.sub(old_pattern, new_code, content, flags=re.MULTILINE | re.DOTALL)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ 代码修改成功！")
else:
    print("❌ 未找到需要替换的代码段")
    # 显示当前代码片段
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "else:" in line and "Keras" in line:
            print(f"第{i+1}行: {line}")
            for j in range(1, 6):
                if i+j < len(lines):
                    print(f"第{i+j+1}行: {lines[i+j]}")
            break
