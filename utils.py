"""
工具函数模块
"""
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from config import LOGGING_CONFIG, OUTPUT_PATHS

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG["filename"], encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_results(data, filename, filepath=None):
    """保存结果到文件"""
    if filepath is None:
        filepath = OUTPUT_PATHS["results"]
    
    os.makedirs(filepath, exist_ok=True)
    full_path = os.path.join(filepath, filename)
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(full_path, index=False, encoding="utf-8")
    elif isinstance(data, np.ndarray):
        np.save(full_path, data)
    else:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(str(data))
    
    print(f"✅ 结果已保存到: {full_path}")

def load_config():
    """加载配置文件"""
    from config import (
        DATA_PATHS, DATA_CONFIG, AUGMENTATION_CONFIG, 
        MODEL_CONFIG, CLASSIFIER_CONFIG, PLOT_CONFIG, 
        OUTPUT_PATHS, CLASS_LABELS, PHASE_LABELS
    )
    return {
        "data_paths": DATA_PATHS,
        "data_config": DATA_CONFIG,
        "augmentation_config": AUGMENTATION_CONFIG,
        "model_config": MODEL_CONFIG,
        "classifier_config": CLASSIFIER_CONFIG,
        "plot_config": PLOT_CONFIG,
        "output_paths": OUTPUT_PATHS,
        "class_labels": CLASS_LABELS,
        "phase_labels": PHASE_LABELS
    }

def ensure_directories():
    """确保所有必要的目录存在"""
    for path in OUTPUT_PATHS.values():
        os.makedirs(path, exist_ok=True)
    print("✅ 所有输出目录已创建")

def get_timestamp():
    """获取当前时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_data_info(X, y, title="数据信息"):
    """打印数据基本信息"""
    print(f"\n📊 {title}")
    print(f"数据形状: {X.shape}")
    print(f"标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"数据类型: {X.dtype}")
    if len(X.shape) == 3:
        print(f"时间步长: {X.shape[1]}")
        print(f"特征维度: {X.shape[2]}")

def validate_data(X, y):
    """验证数据完整性"""
    if len(X) != len(y):
        raise ValueError(f"数据长度不匹配: X={len(X)}, y={len(y)}")
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("数据包含NaN或无穷值")
    
    print("✅ 数据验证通过")
    return True
