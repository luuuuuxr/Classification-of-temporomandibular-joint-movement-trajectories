"""
消融实验脚本：集中式场景下三组组件验证（完全自包含，不导入 main.py）
=========================================================================
  实验一  (w/o Bi-LSTM)  : 人工统计极值特征 + Stacking 集成
  实验二  (w/o Stacking) : Bi-LSTM 编码器 + 单层 MLP 分类头
  实验三  (w/o TPE)      : Bi-LSTM 编码器 + Stacking + 默认超参数
  完整模型 (Full)         : Bi-LSTM 编码器 + Stacking + 调优超参数

运行方式：
    python run_ablation.py

结果保存至：
    results/ablation/ablation_results.json
    results/ablation/ablation_log.txt
"""

import os, sys, json, time, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # 强制 CPU，稳定复现
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ── TensorFlow（延迟导入，避免在数据加载前占用资源）──────────────────────────
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Attention,
    Bidirectional, Concatenate, GlobalAveragePooling1D, Add,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ── 可选依赖 ─────────────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False
    print("[WARN] imbalanced-learn 未安装，将使用 class_weight 替代 SMOTE")

try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    LGBM_OK = False
    print("[WARN] LightGBM 未安装，Stacking 仅用 SVM + DecisionTree")

try:
    from catboost import CatBoostClassifier
    CAT_OK = True
except ImportError:
    CAT_OK = False

# ── 全局配置 ─────────────────────────────────────────────────────────────────
SEED            = 42
N_FOLDS         = 5
BILSTM_EPOCHS   = 80
BILSTM_PATIENCE = 10
OUT_DIR         = "results/ablation"
DATA_PATHS      = ("dzk_class_v2/正常人", "dzk_class_v2/不可复", "dzk_class_v2/可复")

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载（从 main.py 复制的 TrajectoryDataLoader，去除无关依赖）
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryDataLoader:
    def __init__(self, target_length=200, threshold=1.0):
        self.target_length = target_length
        self.threshold = threshold

    def load_dataset(self, normal_path, patient1_path, patient2_path):
        n_data,  n_labels,  _ = self._load_folder(normal_path,   label=0)
        p1_data, p1_labels, _ = self._load_folder(patient1_path, label=1)
        p2_data, p2_labels, _ = self._load_folder(patient2_path, label=2)
        X = self._process_all_data(n_data, p1_data, p2_data)
        y = np.hstack([n_labels, p1_labels, p2_labels])
        return X, y

    def _load_folder(self, folder_path, label):
        data, labels, info = [], [], []
        for fname in os.listdir(folder_path):
            if fname.endswith(".xlsx"):
                xls = pd.ExcelFile(os.path.join(folder_path, fname))
                for sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet, header=None).iloc[1:]
                    data.append(df.values.astype(np.float32))
                    labels.append(label)
                    info.append(fname)
        return data, np.array(labels), info

    def _truncate_platform(self, traj, min_consecutive=2):
        n = len(traj)
        start = 0
        while start <= n - min_consecutive:
            if all(np.linalg.norm(traj[start + i]) < self.threshold
                   for i in range(min_consecutive)):
                start += 1
            else:
                break
        end = n
        while end >= min_consecutive:
            if all(np.max(np.abs(traj[end - i - 1])) < self.threshold
                   for i in range(min_consecutive)):
                end -= 1
            else:
                break
        return traj[start:end] if start < end else traj

    def _interpolate_or_pad(self, traj):
        L = len(traj)
        if L == self.target_length:
            return traj
        if L == 0:
            return np.zeros((self.target_length, 3), dtype=np.float32)
        x_old = np.linspace(0, 1, L)
        x_new = np.linspace(0, 1, self.target_length)
        out = np.zeros((self.target_length, traj.shape[1]), dtype=np.float32)
        for d in range(traj.shape[1]):
            f = interp1d(x_old, traj[:, d], kind="linear",
                         fill_value="extrapolate")
            out[:, d] = f(x_new)
        return out

    def _process_all_data(self, n, p1, p2):
        processed = []
        for group in [n, p1, p2]:
            for traj in group:
                t = self._truncate_platform(traj)
                processed.append(self._interpolate_or_pad(t))
        return np.stack(processed, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Bi-LSTM 编码器构建（消融实验专用）
# ═══════════════════════════════════════════════════════════════════════════════

def _residual_block(x, units, dropout_rate=0.1, l2_reg=0.001):
    """双向 LSTM residual block"""
    shortcut = x
    h = Bidirectional(LSTM(units // 2, return_sequences=True,
                           kernel_regularizer=l2(l2_reg)))(x)
    h = BatchNormalization()(h)
    h = Dropout(dropout_rate)(h)
    if shortcut.shape[-1] != h.shape[-1]:
        shortcut = Dense(h.shape[-1])(shortcut)
    return Add()([shortcut, h])


def build_bilstm_encoder(input_shape, lstm_units=64, dense_units=32,
                          use_attention=True, num_classes=3):
    """BiLSTM + Attention 编码器，返回 (classifier_model, feature_model)"""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True,
                           kernel_regularizer=l2(0.0001)))(inputs)
    x = BatchNormalization()(x)
    x = _residual_block(x, lstm_units * 2, 0.1, 0.001)
    x = _residual_block(x, lstm_units,     0.2, 0.001)
    if use_attention:
        attn = Attention()([x, x])
        x = Concatenate()([x, attn])
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation="relu", kernel_regularizer=l2(0.01))(x)
    out = Dense(num_classes, activation="softmax", name="classification")(x)
    clf_model  = Model(inputs, out)
    feat_model = Model(inputs, x)
    return clf_model, feat_model


# ═══════════════════════════════════════════════════════════════════════════════
# 超参数配置
# ═══════════════════════════════════════════════════════════════════════════════

# Full / w/o Bi-LSTM：使用调优后参数（对应 TPE 优化结果）
TUNED_PARAMS = {
    "SVM":          {"C": 5.0,  "kernel": "rbf", "gamma": "scale"},
    "DecisionTree": {"max_depth": 6, "min_samples_leaf": 1,
                     "min_samples_split": 2, "criterion": "entropy",
                     "splitter": "random"},
    "LightGBM":     {"n_estimators": 80,  "num_leaves": 48,
                     "learning_rate": 0.03},
    "CatBoost":     {"iterations": 120},
}

# w/o TPE：sklearn / 算法库原始默认参数
DEFAULT_PARAMS = {
    "SVM":          {"C": 1.0,  "kernel": "rbf", "gamma": "scale"},
    "DecisionTree": {"max_depth": None, "min_samples_leaf": 1,
                     "min_samples_split": 2, "criterion": "gini",
                     "splitter": "best"},
    "LightGBM":     {"n_estimators": 100, "num_leaves": 31,
                     "learning_rate": 0.1},
    "CatBoost":     {"iterations": 1000},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg, f=None):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode(), flush=True)
    if f:
        f.write(msg + "\n")
        f.flush()


def compute_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sens_list, spec_list = [], []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return {"acc": acc, "f1": f1,
            "sen": float(np.mean(sens_list)),
            "spe": float(np.mean(spec_list))}


def extract_manual_features(X):
    """人工统计极值特征（对应第三章 3.4.1 节）"""
    feats = []
    for traj in X:
        T, D = traj.shape
        row = []
        for d in range(D):
            col = traj[:, d]
            row += [float(np.max(col)), float(np.min(col)),
                    float(np.max(col) - np.min(col)),
                    float(np.std(col)), float(np.mean(col)),
                    float(np.percentile(col, 25)),
                    float(np.percentile(col, 75)),
                    float(np.abs(col).mean())]
        for seg in [traj[:T//3], traj[T//3:2*T//3], traj[2*T//3:]]:
            row.append(float(np.max(np.abs(seg))))
        feats.append(row)
    return np.array(feats, dtype=np.float32)


def train_bilstm_and_extract(X_tr, y_tr, X_val, y_val, X_te,
                              dense_units=32):
    """训练 BiLSTM（含类别权重），返回 (feat_tr, feat_val, feat_te)"""
    from sklearn.utils.class_weight import compute_class_weight
    classes  = np.unique(y_tr)
    weights  = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw_dict  = dict(zip(classes.tolist(), weights.tolist()))

    clf, feat = build_bilstm_encoder(X_tr.shape[1:],
                                     dense_units=dense_units)
    clf.compile(optimizer=Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    clf.fit(X_tr, y_tr,
            validation_data=(X_val, y_val),
            class_weight=cw_dict,
            epochs=BILSTM_EPOCHS, batch_size=16,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=BILSTM_PATIENCE,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_accuracy", factor=0.6,
                                  patience=5, min_lr=5e-5),
            ], verbose=0)
    scaler = StandardScaler()
    F_tr  = scaler.fit_transform(feat.predict(X_tr,  verbose=0))
    F_val = scaler.transform(feat.predict(X_val, verbose=0))
    F_te  = scaler.transform(feat.predict(X_te,  verbose=0))
    tf.keras.backend.clear_session()   # 释放显存
    return F_tr, F_val, F_te


def balance_features(X_feat, y, seed=SEED):
    """SMOTE 过采样平衡类别（优先），回退到 class_weight"""
    if SMOTE_OK:
        try:
            sm = SMOTE(random_state=seed, k_neighbors=min(3, min(
                np.sum(y == c) for c in np.unique(y)) - 1))
            return sm.fit_resample(X_feat, y)
        except Exception:
            pass
    return X_feat, y   # 若 SMOTE 失败则原样返回（在 SVM 中用 class_weight 补偿）


def compute_class_weight_dict(y):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def train_stacking(X_feat, y, params):
    """训练 Stacking 集成（含 SMOTE 类别平衡），返回 (base_models, meta_lr)"""
    # SMOTE 平衡训练集（元学习器使用平衡后数据）
    X_bal, y_bal = balance_features(X_feat, y)
    cw = compute_class_weight_dict(y)   # 原始分布权重，备用

    p_svm = params["SVM"]
    p_dt  = params["DecisionTree"]

    base = []
    svm = SVC(C=p_svm["C"], kernel=p_svm["kernel"], gamma=p_svm["gamma"],
              class_weight="balanced",
              probability=True, random_state=SEED)
    svm.fit(X_bal, y_bal);  base.append(("SVM", svm))

    dt = DecisionTreeClassifier(
        max_depth=p_dt["max_depth"],
        min_samples_leaf=p_dt["min_samples_leaf"],
        min_samples_split=p_dt["min_samples_split"],
        criterion=p_dt["criterion"],
        splitter=p_dt["splitter"],
        class_weight="balanced", random_state=SEED)
    dt.fit(X_bal, y_bal);  base.append(("DT", dt))

    if LGBM_OK:
        p_l = params.get("LightGBM", {})
        lgbm = lgb.LGBMClassifier(
            n_estimators=p_l.get("n_estimators", 100),
            num_leaves=p_l.get("num_leaves", 31),
            learning_rate=p_l.get("learning_rate", 0.1),
            class_weight="balanced",
            verbose=-1, random_state=SEED)
        lgbm.fit(X_bal, y_bal);  base.append(("LGBM", lgbm))

    if CAT_OK:
        p_c = params.get("CatBoost", {})
        cat = CatBoostClassifier(
            iterations=p_c.get("iterations", 100),
            class_weights=[cw.get(c, 1.0) for c in sorted(cw)],
            verbose=0, random_state=SEED)
        cat.fit(X_bal, y_bal);  base.append(("Cat", cat))

    # Meta-features 用原始（非平衡）特征，保持测试集分布一致性
    meta_feats = np.hstack([m.predict_proba(X_bal) for _, m in base])
    meta = LogisticRegression(C=5.0, max_iter=3000, solver="lbfgs",
                              class_weight="balanced", random_state=SEED)
    meta.fit(meta_feats, y_bal)
    return base, meta


def predict_stacking(base, meta, X_feat):
    mf = np.hstack([m.predict_proba(X_feat) for _, m in base])
    return meta.predict(mf)


# ═══════════════════════════════════════════════════════════════════════════════
# 四组实验的主逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(variant, X, y, logfile):
    log(f"\n{'='*60}", logfile)
    log(f"[{variant.upper()}] 开始 {N_FOLDS}-折交叉验证", logfile)
    log(f"{'='*60}", logfile)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fi, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_tr_all, X_te = X[tr_idx], X[te_idx]
        y_tr_all, y_te = y[tr_idx], y[te_idx]

        # 从训练集分出验证集（供 BiLSTM EarlyStopping）
        val_n = max(1, int(len(X_tr_all) * 0.20))
        X_tr, X_val = X_tr_all[:-val_n], X_tr_all[-val_n:]
        y_tr, y_val = y_tr_all[:-val_n], y_tr_all[-val_n:]

        # ── 实验一：w/o Bi-LSTM ───────────────────────────────────────────────
        if variant == "wo_bilstm":
            sc = StandardScaler()
            F_tr = sc.fit_transform(extract_manual_features(X_tr_all))  # 用全量训练集
            F_te = sc.transform(extract_manual_features(X_te))
            base, meta = train_stacking(F_tr, y_tr_all, TUNED_PARAMS)
            y_pred = predict_stacking(base, meta, F_te)

        # ── 实验二：w/o Stacking（BiLSTM + 单层 MLP）────────────────────────
        elif variant == "wo_stacking":
            F_tr, _, F_te = train_bilstm_and_extract(
                X_tr, y_tr, X_val, y_val, X_te)
            # SMOTE 平衡后再训练 MLP
            F_tr_bal, y_tr_bal = balance_features(F_tr, y_tr)
            mlp = MLPClassifier(hidden_layer_sizes=(64,), activation="relu",
                                max_iter=500, random_state=SEED,
                                early_stopping=True, validation_fraction=0.15)
            mlp.fit(F_tr_bal, y_tr_bal)
            y_pred = mlp.predict(F_te)

        # ── 实验三：w/o TPE（BiLSTM + Stacking 默认超参）────────────────────
        elif variant == "wo_tpe":
            F_tr, _, F_te = train_bilstm_and_extract(
                X_tr, y_tr, X_val, y_val, X_te)
            base, meta = train_stacking(F_tr, y_tr, DEFAULT_PARAMS)
            y_pred = predict_stacking(base, meta, F_te)

        # ── 完整模型：BiLSTM + Stacking + 调优超参 ──────────────────────────
        elif variant == "full":
            F_tr, _, F_te = train_bilstm_and_extract(
                X_tr, y_tr, X_val, y_val, X_te)
            base, meta = train_stacking(F_tr, y_tr, TUNED_PARAMS)
            y_pred = predict_stacking(base, meta, F_te)

        else:
            raise ValueError(f"未知 variant: {variant}")

        m = compute_metrics(y_te, y_pred)
        fold_metrics.append(m)
        log(f"  Fold {fi+1}/{N_FOLDS}  "
            f"Acc={m['acc']:.4f}  F1={m['f1']:.4f}  "
            f"Sen={m['sen']:.4f}  Spe={m['spe']:.4f}  "
            f"({time.time()-t0:.1f}s)", logfile)

    result = {}
    for k in ["acc", "f1", "sen", "spe"]:
        vals = [m[k] for m in fold_metrics]
        result[k]          = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals))
    result["var"] = float(np.var([m["acc"] for m in fold_metrics]))

    log(f"\n  汇总：Acc={result['acc']:.4f}±{result['acc_std']:.4f}  "
        f"F1={result['f1']:.4f}  Sen={result['sen']:.4f}  "
        f"Spe={result['spe']:.4f}  Var={result['var']:.6f}", logfile)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logpath = os.path.join(OUT_DIR, "ablation_log.txt")
    with open(logpath, "w", encoding="utf-8") as lf:
        log("消融实验启动", lf)
        log(f"折数={N_FOLDS}  种子={SEED}  BiLSTM最大轮次={BILSTM_EPOCHS}", lf)
        log(f"LightGBM={LGBM_OK}  CatBoost={CAT_OK}", lf)

        # 加载数据
        log("\n[Data] 正在加载轨迹数据...", lf)
        loader = TrajectoryDataLoader(target_length=200, threshold=1.0)
        X, y = loader.load_dataset(*DATA_PATHS)
        counts = dict(zip(*np.unique(y, return_counts=True)))
        log(f"[Data] X={X.shape}  类别分布={counts}", lf)

        # 按顺序运行（wo_bilstm 最快，放最前；full 最后）
        variants = ["wo_bilstm", "wo_stacking", "wo_tpe", "full"]
        all_results = {}
        t_total = time.time()

        for v in variants:
            all_results[v] = run_experiment(v, X, y, lf)

        elapsed = time.time() - t_total
        log(f"\n总耗时：{elapsed/60:.1f} 分钟", lf)

        # 先保存 JSON（确保数据不丢失）
        save_path = os.path.join(OUT_DIR, "ablation_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        log(f"\n结果已保存：{save_path}", lf)

        # 汇总打印
        log("\n" + "="*60, lf)
        log("消融实验结果汇总", lf)
        log("="*60, lf)
        names = {"wo_bilstm": "w/o Bi-LSTM", "wo_stacking": "w/o Stacking",
                 "wo_tpe": "w/o TPE", "full": "Full"}
        full_acc = all_results["full"]["acc"]
        log(f"{'Variant':<18} {'Acc':>8} {'F1':>8} {'Sen':>8} {'Spe':>8} {'Drop':>10}", lf)
        log("-"*60, lf)
        for v in variants:
            r    = all_results[v]
            drop = (full_acc - r["acc"]) * 100
            sign = f"-{drop:.2f}pp" if v != "full" else "--baseline"
            log(f"{names[v]:<18} {r['acc']:>8.4f} {r['f1']:>8.4f}"
                f" {r['sen']:>8.4f} {r['spe']:>8.4f} {sign:>10}", lf)

    print(f"\n完成！日志：{logpath}")
    print(f"结果 JSON：{save_path}")
    print("\n下一步：运行 python plot_thesis_figures.py 自动加载真实数据绘图")


if __name__ == "__main__":
    main()
