import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── GPU 运行模式开关 ────────────────────────────────────────────────────────
# USE_GPU = True  → 尝试使用 GPU（需要 TF 与 cuDNN 版本完全匹配）
# USE_GPU = False → 强制 CPU（避免 CUDNN_STATUS_INTERNAL_ERROR）
USE_GPU = False
# ────────────────────────────────────────────────────────────────────────────

if USE_GPU:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import warnings
import logging
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*retracing.*", category=UserWarning)

tf.random.set_seed(42)

try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    pass

import seaborn as sns
sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)

# ── 新模块化结构导入 ──────────────────────────────────────────────────────────
from pipelines.standard import run_standard_pipeline
from pipelines.ablation import run_ablation_pipeline


def main():
    # ======================================================
    #  运行模式开关
    #  RUN_ABLATION  : 消融实验（第四章，对比 w/o Bi-LSTM / w/o Stacking / w/o TPE）
    #  False         : 集中式标准流程（第四章）
    # ======================================================
    RUN_ABLATION  = False    # 消融实验（改为 True 即可运行）

    if RUN_ABLATION:
        # N_TRIALS：每种配置重复实验次数（不同随机种子），建议 5~10 次
        run_ablation_pipeline(n_trials=5)
        return

    run_standard_pipeline()


if __name__ == '__main__':
    main()

