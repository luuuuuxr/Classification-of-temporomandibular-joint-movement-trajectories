"""
主程序 - 颞下颌关节疾病分类系统
优化后的版本，代码结构更清晰
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# 导入自定义模块
from config import DATA_PATHS, DATA_CONFIG, CLASSIFIER_CONFIG
from utils import setup_logging, ensure_directories, print_data_info
from data_processing import TrajectoryDataLoader, data_split_and_augment
from model_builder import lstm_time_info_extract, evaluate_model, summarize_results
from trajectory_build import split_into_phases, plot_phases_3d
from model_explainability import plot_phase_shap_importance, plot_phase_aggregated_feature_importance

# 设置随机种子
tf.random.set_seed(42)

def main():
    """主函数"""
    # 设置日志和目录
    logger = setup_logging()
    ensure_directories()
    
    logger.info("🚀 开始颞下颌关节疾病分类分析")
    
    # ============ 数据加载 ============
    logger.info("📁 正在加载数据...")
    loader = TrajectoryDataLoader(
        target_length=DATA_CONFIG["target_length"],
        threshold=DATA_CONFIG["threshold"]
    )
    
    X, X_processed, y, file_info = loader.load_dataset(
        DATA_PATHS["normal"],
        DATA_PATHS["reversible"], 
        DATA_PATHS["irreversible"]
    )
    
    # 使用处理后的数据
    X = X_processed
    
    # 确保数据类型正确
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print_data_info(X, y, "数据集")
    
    # ============ 数据分割和增强 ============
    logger.info("🔄 正在进行数据分割和增强...")
    X_train, y_train, X_train_resampled, y_train_resampled, X_val, X_test, y_val, y_test = data_split_and_augment(
        X, y, 
        test_size=DATA_CONFIG["test_size"],
        val_size=DATA_CONFIG["val_size"],
        random_state=DATA_CONFIG["random_state"]
    )
    
    X_train = X_train_resampled
    y_train = y_train_resampled
    
    # ============ 特征提取 ============
    logger.info("🔍 正在进行LSTM特征提取...")
    X_train_features, X_test_features = lstm_time_info_extract(X_train, y_train, X_val, y_val, X_test)
    # X_test_features 已从 lstm_time_info_extract 获取
    
    # ============ 模型训练 ============
    logger.info("🤖 正在训练分类器...")
    
    # 创建分类器
    svm_model = SVC(**CLASSIFIER_CONFIG["svm"])
    rf_model = RandomForestClassifier(**CLASSIFIER_CONFIG["random_forest"])
    dt_model = DecisionTreeClassifier(**CLASSIFIER_CONFIG["decision_tree"])
    
    # 训练模型
    svm_model.fit(X_train_features, y_train)
    rf_model.fit(X_train_features, y_train)
    dt_model.fit(X_train_features, y_train)
    
    # ============ 模型评估 ============
    logger.info("📊 正在评估模型性能...")
    
    result_svm = evaluate_model(svm_model, X_train_features, y_train, X_test_features, y_test, "SVM", file_info)
    result_rf = evaluate_model(rf_model, X_train_features, y_train, X_test_features, y_test, "RandomForest", file_info)
    result_dt = evaluate_model(dt_model, X_train_features, y_train, X_test_features, y_test, "DecisionTree", file_info)
    
    # 保存结果摘要
    results_dict = {
        "SVM": result_svm,
        "RandomForest": result_rf,
        "DecisionTree": result_dt,
    }
    
    summary_df = summarize_results(results_dict)
    summary_df.to_csv("results/model_summary.csv", index=False)
    
    # ============ 可解释性分析 ============
    logger.info("🔬 正在进行可解释性分析...")
    
    # 阶段分割
    all_phases = [split_into_phases(trajectory) for trajectory in X_test]
    num_phases = 6
    
    for phases in all_phases:
        assert len(phases) == num_phases, f"Expected {num_phases} phases but got {len(phases)}"
    
    # 绘制阶段3D图
    plot_phases_3d(all_phases, y_test, file_info, base_save_path="3D_Trajectories_new_Phases", num_phases=num_phases)
    
    # SHAP分析 - 修复特征索引问题
    # LSTM特征维度是32，需要按比例分配给6个阶段
    feature_dim = X_test_features.shape[1]  # 应该是32
    phases_by_stage = []
    
    # 将32个特征按比例分配给6个阶段
    features_per_phase = feature_dim // num_phases
    remainder = feature_dim % num_phases
    
    start_idx = 0
    for i in range(num_phases):
        # 计算该阶段的特征数量
        phase_size = features_per_phase + (1 if i < remainder else 0)
        end_idx = start_idx + phase_size
        
        # 生成该阶段的特征索引
        phase_indices = list(range(start_idx, end_idx))
        phases_by_stage.append(phase_indices)
        
        start_idx = end_idx
    
    print(f'特征维度: {feature_dim}, 阶段分配: {[len(phase) for phase in phases_by_stage]}')
    
    models = [svm_model, rf_model, dt_model]
    model_names = ["SVM", "RandomForest", "DecisionTree"]
    phases_name = [f"Phase {i+1}" for i in range(num_phases)]
    
    shap_values_list = plot_phase_shap_importance(
        models, model_names, X_train_features, y_train, X_test_features, phases_by_stage
    )
    
    plot_phase_aggregated_feature_importance(shap_values_list, phases_name)
    
    logger.info("✅ 分析完成！结果已保存到results目录")

if __name__ == "__main__":
    main()
