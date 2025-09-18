"""
配置文件 - 包含所有常量和参数设置
"""
import os

# 数据路径配置
DATA_PATHS = {
    "normal": "dzk_class_v2/正常人",
    "reversible": "dzk_class_v2/可复", 
    "irreversible": "dzk_class_v2/不可复"
}

# 数据预处理参数
DATA_CONFIG = {
    "target_length": 200,
    "threshold": 1.0,
    "min_consecutive": 2,
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42
}

# 数据增强参数
AUGMENTATION_CONFIG = {
    "target_counts": {0: 100, 1: 50, 2: 50},
    "noise_std": 0.1,
    "rotation_angle": 5,
    "time_warp_factor": 0.1
}

# 模型参数
MODEL_CONFIG = {
    "lstm_units": [64, 32, 16],
    "dense_units": 32,
    "dropout_rate": 0.3,
    "l2_reg": 0.01,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 100,
    "patience": 7
}

# 分类器参数
CLASSIFIER_CONFIG = {
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": 42
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "decision_tree": {
        "max_depth": 10,
        "random_state": 42
    }
}

# 可视化参数
PLOT_CONFIG = {
    "figsize": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
}

# 输出路径配置
OUTPUT_PATHS = {
    "results": "results",
    "models": "results/models",
    "plots": "results/plots",
    "logs": "results/logs",
    "trajectories_original": "3D_Trajectories_original",
    "trajectories_processed": "3D_Trajectories_processed",
    "trajectories_phases": "3D_Trajectories_new_Phases",
    "plots_2d_original": "2D_Plots_original",
    "plots_2d_processed": "2D_Plots_processed"
}

# 类别标签
CLASS_LABELS = {
    0: "Normal",
    1: "DDwoR", 
    2: "DDwR"
}

# 阶段标签
PHASE_LABELS = [f"Phase {i+1}" for i in range(6)]

# 确保输出目录存在
def ensure_output_dirs():
    """确保所有输出目录存在"""
    for path in OUTPUT_PATHS.values():
        os.makedirs(path, exist_ok=True)

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": "results/logs/trajectory_analysis.log"
}
