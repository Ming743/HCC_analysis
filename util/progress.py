import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def build_CO_vector(df_all: pd.DataFrame,
                    sample_feature_cols: list,
                    t_list=None,
                    n_seeds: int = 100,
                    test_size: float = 0.2,
                    config_path: str = "config.json",
                    base_cls_params: dict = None):
    """
    每個 seed：
      - 9 個分類器：只把「驗證集」機率寫入 outputs_val_only（train 不寫）
      - 最後：對每個 cell，若有驗證值就取平均，否則保持 NaN

    參數
    ----
    df_all : 包含至少 ['time','event'] + sample_feature_cols 的 DataFrame
    sample_feature_cols : 特徵欄位清單
    t_list : 要跑的時間閾值清單；若為 None，預設使用 config.json 內 CO_params 的 key（排序後）
    n_seeds : 重複次數（不同 random_state）
    test_size : 驗證集比例
    config_path : config.json 路徑（需包含 {"CO_params": {...}}）
    base_cls_params : 給 XGBClassifier 的共用基礎參數（可覆蓋預設值）
                      預設為:
                        {
                          "objective": "binary:logistic",
                          "eval_metric": "logloss",
                          "tree_method": "hist",
                          "verbosity": 0,
                          "device": "cuda"
                        }
                      若無 GPU 可改成 device="cpu"
    回傳
    ----
    final : DataFrame，index 與 df_all 對齊，欄位為 [f"T={t}" for t in t_list]，
            值為 OOF（驗證集）機率的多次平均；若從未被抽中驗證集則為 NaN。
    """

    # 讀 config.json 並取出 CO_params
    with open(config_path, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

    if "CO_params" not in _cfg:
        raise KeyError(f"`CO_params` not found in {config_path}")

    # 將 key 轉成 float 方便用數值比較/排序
    CO_params = {float(k): v for k, v in _cfg["CO_params"].items()}

    # 若沒給 t_list，就用 config 裡的 key（排序）
    if t_list is None:
        t_list = sorted(CO_params.keys())

    cols_9T = [f"T={t}" for t in t_list]

    # 預設的 XGBClassifier 基礎參數
    default_base = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
        "device": "cuda"   # 沒 GPU 的話改成 "cpu"
    }
    if base_cls_params is not None:
        default_base.update(base_cls_params)

    # 全局累積器：只累計驗證集格子
    val_sum = pd.DataFrame(0.0, index=df_all.index, columns=cols_9T)
    val_cnt = pd.DataFrame(0,   index=df_all.index, columns=cols_9T, dtype=int)

    for seed in tqdm(range(n_seeds), desc="Seeds"):
        # 這份只放「驗證集」的機率
        outputs_val_only = pd.DataFrame(np.nan, index=df_all.index, columns=cols_9T, dtype=float)

        # === 逐 T 訓練二分類器 ===
        for j, T in enumerate(t_list):
            col = cols_9T[j]

            # 排除 (time <= T & event==0)
            mask_keep = ~((df_all["time"] <= T) & (df_all["event"] == 0))
            valid = df_all[mask_keep].copy()

            # label
            valid["label"] = (valid["time"] >= T).astype(int)
            X_all = valid[sample_feature_cols]
            y_all = valid["label"].astype(int)
            idx_all = valid.index

            X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(
                X_all, y_all, idx_all, test_size=test_size, random_state=seed, stratify=y_all
            )

            # 合併：基礎參數 + 該 T 的超參數
            if T not in CO_params:
                raise KeyError(f"T={T} not found in CO_params loaded from {config_path}")
            this_params = default_base.copy()
            this_params.update(CO_params[T])

            clf = xgb.XGBClassifier(**this_params)
            clf.fit(X_tr, y_tr)

            prob_va = clf.predict_proba(X_va)[:, 1]

            # 只把驗證集寫入最終平均用矩陣（train 保持 NaN）
            outputs_val_only.loc[idx_va, col] = prob_va

        # === 累積：只累計驗證值 ===
        val_mask = outputs_val_only.notna()
        val_sum[val_mask] += outputs_val_only[val_mask]
        val_cnt[val_mask] += 1

    # === 匯總：僅使用驗證平均，無驗證就留 NaN ===
    final = pd.DataFrame(np.nan, index=df_all.index, columns=cols_9T, dtype=float)
    has_val = (val_cnt.values > 0)
    final.values[has_val] = (val_sum.values[has_val] / np.maximum(val_cnt.values[has_val], 1))

    return final
