"""
3D轨迹架构对比实验主脚本
比较简单LSTM、增强LSTM+3D特征、传统ML+3D特征三种方案
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from trajectory_3d_processor import compare_3d_architectures, Trajectory3DProcessor
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_sample_data():
    """加载示例数据（如果没有真实数据，生成模拟数据）"""
    print("生成模拟3D轨迹数据...")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 300
    time_steps = 200
    
    # 正常组：平滑的3D轨迹
    normal_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        x = np.sin(t) + 0.1 * np.random.randn(time_steps)
        y = np.cos(t) + 0.1 * np.random.randn(time_steps)
        z = 0.5 * t + 0.1 * np.random.randn(time_steps)
        trajectory = np.column_stack([x, y, z])
        normal_data.append(trajectory)
    
    # 患者组1：不规则的3D轨迹
    patient1_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        x = np.sin(t) + 0.3 * np.random.randn(time_steps) + 0.2 * np.sin(3*t)
        y = np.cos(t) + 0.3 * np.random.randn(time_steps) + 0.2 * np.cos(3*t)
        z = 0.5 * t + 0.3 * np.random.randn(time_steps) + 0.1 * np.sin(2*t)
        trajectory = np.column_stack([x, y, z])
        patient1_data.append(trajectory)
    
    # 患者组2：更复杂的3D轨迹
    patient2_data = []
    for i in range(n_samples // 3):
        t = np.linspace(0, 4*np.pi, time_steps)
        x = np.sin(2*t) + 0.4 * np.random.randn(time_steps) + 0.3 * np.sin(5*t)
        y = np.cos(2*t) + 0.4 * np.random.randn(time_steps) + 0.3 * np.cos(5*t)
        z = 0.3 * t + 0.4 * np.random.randn(time_steps) + 0.2 * np.sin(3*t)
        trajectory = np.column_stack([x, y, z])
        patient2_data.append(trajectory)
    
    # 合并数据
    X = np.array(normal_data + patient1_data + patient2_data)
    y = np.array([0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3))
    
    return X, y

def load_real_data():
    """尝试加载真实数据"""
    try:
        # 尝试导入现有的数据加载器
        from main import TrajectoryDataLoader
        
        # 设置数据路径（请根据实际情况修改）
        normal_path = "data/normal"
        patient1_path = "data/patient1" 
        patient2_path = "data/patient2"
        
        # 检查路径是否存在
        if os.path.exists(normal_path) and os.path.exists(patient1_path) and os.path.exists(patient2_path):
            print("加载真实数据...")
            loader = TrajectoryDataLoader(target_length=200, threshold=1.0)
            X_raw, X_processed, y, file_info = loader.load_dataset(normal_path, patient1_path, patient2_path)
            return X_processed, y
        else:
            print("真实数据路径不存在，使用模拟数据...")
            return load_sample_data()
    except ImportError:
        print("无法导入数据加载器，使用模拟数据...")
        return load_sample_data()
    except Exception as e:
        print(f"加载真实数据时出错: {e}")
        print("使用模拟数据...")
        return load_sample_data()

def main():
    """主实验函数"""
    print("=" * 80)
    print("3D轨迹处理架构对比实验")
    print("=" * 80)
    
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 加载数据
    X, y = load_real_data()
    
    print(f"数据形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 数据预处理
    print("\n数据预处理...")
    
    # 标准化数据
    X_reshaped = X.reshape(X.shape[0], -1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    # 分割数据
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 运行对比实验
    print("\n开始架构对比实验...")
    input_shape = (X.shape[1], X.shape[2])  # (200, 3)
    
    results, histories = compare_3d_architectures(
        X_train, y_train, X_val, y_val, input_shape, save_path="results"
    )
    
    # 在测试集上评估最佳模型
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型...")
    print("=" * 60)
    
    # 找到最佳模型
    best_model_name = max(results, key=results.get)
    print(f"最佳模型: {best_model_name}")
    print(f"验证准确率: {results[best_model_name]:.4f}")
    
    # 在测试集上评估
    if "Enhanced LSTM" in best_model_name:
        # 使用增强LSTM
        processor = Trajectory3DProcessor(input_shape)
        processor.build_enhanced_lstm_model()
        
        # 重新训练（使用全部训练+验证数据）
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        
        spatial_features, _, _ = processor.preprocess_trajectories(X_train_val)
        spatial_features_test, _, _ = processor.preprocess_trajectories(X_test)
        
        processor.model.fit(
            [X_train_val, spatial_features], y_train_val,
            epochs=30, batch_size=32, verbose=1
        )
        
        test_pred = processor.predict(X_test, use_spatial_features=True)
    elif "Simple LSTM" in best_model_name:
        # 使用简单LSTM
        processor = Trajectory3DProcessor(input_shape)
        processor.build_simple_lstm_model()
        
        # 重新训练
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        
        processor.model.fit(
            X_train_val, y_train_val,
            epochs=30, batch_size=32, verbose=1
        )
        
        test_pred = processor.predict(X_test, use_spatial_features=False)
    else:
        # 使用传统ML方法
        processor = Trajectory3DProcessor(input_shape)
        spatial_features_train_val, _, _ = processor.preprocess_trajectories(np.vstack([X_train, X_val]))
        spatial_features_test, _, _ = processor.preprocess_trajectories(X_test)
        y_train_val = np.hstack([y_train, y_val])
        
        if "Random Forest" in best_model_name:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.svm import SVC
            model = SVC(kernel="rbf", random_state=42)
        
        model.fit(spatial_features_train_val, y_train_val)
        test_pred = model.predict(spatial_features_test)
    
    # 计算测试集准确率
    if len(test_pred.shape) > 1:  # 如果是概率输出
        test_pred_labels = np.argmax(test_pred, axis=1)
    else:
        test_pred_labels = test_pred
    
    from sklearn.metrics import accuracy_score, classification_report
    test_accuracy = accuracy_score(y_test, test_pred_labels)
    
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 保存测试结果
    with open("results/test_results.txt", "w", encoding="utf-8") as f:
        f.write("测试集评估结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"最佳模型: {best_model_name}\n")
        f.write(f"验证准确率: {results[best_model_name]:.4f}\n")
        f.write(f"测试准确率: {test_accuracy:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(classification_report(y_test, test_pred_labels, 
                                    target_names=["Normal", "Patient1", "Patient2"]))
    
    print("\n实验完成！")
    print("所有结果已保存到 results 文件夹")
    
    return results, histories

if __name__ == "__main__":
    results, histories = main()
