# 运动轨迹分析系统

## 项目简介
基于深度学习的下颌运动轨迹分析系统，用于识别正常人和颞下颌关节紊乱患者的运动模式差异。

## 主要功能
- 轨迹数据预处理和增强
- LSTM特征提取
- 多种分类器训练（SVM、RandomForest、DecisionTree、CatBoost等）
- 模型评估和可视化
- 可解释性分析

## 文件结构
\\\
├── main.py                    # 主程序入口
├── config.py                  # 配置文件
├── src/                       # 源代码模块
├── dzk_class_v2/             # 原始数据目录
│   ├── 正常人/               # 正常人轨迹数据
│   ├── 不可复/               # 不可复性关节盘移位数据
│   └── 可复/                 # 可复性关节盘移位数据
├── results/                   # 分析结果
├── 3D_Trajectories_new_Phases/ # 阶段化3D轨迹图
├── best_model.h5             # 训练好的LSTM模型
├── enhanced_lstm_model.h5    # 增强LSTM模型
├── model_summary.csv         # 模型性能汇总
└── augmentation_log.xlsx     # 数据增强日志
\\\

## 运行方法
\\\ash
python main.py
\\\

## 主要输出
- 模型性能评估报告
- 混淆矩阵和ROC曲线
- 3D轨迹可视化
- 阶段化分析结果
- 可解释性分析图表

## 依赖环境
- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- 其他依赖见代码中的import语句
