import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib

# 使用无界面后端保存图像
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_PATH = Path(__file__).resolve().parent / "cluster_df.csv"
RAW_PATH = Path(__file__).resolve().parent / "raw_df.csv"
OUTPUT_FEATURES = Path(__file__).resolve().parent / "kmeans_features.csv"
OUTPUT_EVAL = Path(__file__).resolve().parent / "kmeans_eval_results.csv"
OUTPUT_ELBOW = Path(__file__).resolve().parent / "kmeans_elbow.png"
OUTPUT_SIL = Path(__file__).resolve().parent / "kmeans_silhouette.png"
OUTPUT_LABELS = Path(__file__).resolve().parent / "cluster_labels.csv"
OUTPUT_MEAN_RES = Path(__file__).resolve().parent / "cluster_feature_means_resources.png"
OUTPUT_MEAN_DIG = Path(__file__).resolve().parent / "cluster_feature_means_digital.png"
OUTPUT_MEAN_FIN = Path(__file__).resolve().parent / "cluster_feature_means_financial.png"
OUTPUT_MEAN_HE = Path(__file__).resolve().parent / "cluster_feature_means_health_env.png"
OUTPUT_CROSSTAB = Path(__file__).resolve().parent / "cluster_coop_crosstab.csv"
OUTPUT_CLUSTER_SUMMARY = Path(__file__).resolve().parent / "cluster_income_satisfaction.csv"

READ_ENCODING = "utf-8-sig"
WRITE_ENCODING = "utf-8-sig"
MERGE_KEY = "varID"

# 高相关与近零方差阈值
CORR_THRESHOLD = 0.95
NEAR_ZERO_VAR = 1e-8


def detect_binary_and_numeric(feature_df: pd.DataFrame):
    """自动区分 0/1 二元与连续特征。"""
    binary_features = []
    numeric_features = []
    for col in feature_df.columns:
        uniq = feature_df[col].dropna().unique()
        uniq_set = set(pd.to_numeric(pd.Series(list(uniq)), errors="coerce").dropna().tolist())
        if len(uniq_set) <= 2 and uniq_set.issubset({0, 1}):
            binary_features.append(col)
        else:
            numeric_features.append(col)
    print(f"检测到二元特征 ({len(binary_features)}): {binary_features}")
    print(f"检测到连续特征 ({len(numeric_features)}): {numeric_features}")
    return numeric_features, binary_features


def drop_high_corr(numeric_cols: list[str], feature_df: pd.DataFrame):
    """剔除高度相关(>|0.95|)的重复连续特征，优先保留含 total 的列。"""
    corr = feature_df[numeric_cols].corr().abs()
    to_drop = set()
    order_map = {c: i for i, c in enumerate(numeric_cols)}
    for i, col_a in enumerate(numeric_cols):
        for j, col_b in enumerate(numeric_cols):
            if j <= i:
                continue
            if corr.loc[col_a, col_b] > CORR_THRESHOLD:
                a_total = "total" in col_a.lower()
                b_total = "total" in col_b.lower()
                if a_total and not b_total:
                    keep, drop = col_a, col_b
                elif b_total and not a_total:
                    keep, drop = col_b, col_a
                else:
                    keep, drop = (col_a, col_b) if order_map[col_a] < order_map[col_b] else (col_b, col_a)
                print(f"因高相关剔除 {drop}，保留 {keep}（|r|={corr.loc[col_a, col_b]:.3f}）")
                to_drop.add(drop)
    final_cols = [c for c in numeric_cols if c not in to_drop]
    print(f"相关性筛选前连续特征数: {len(numeric_cols)}，筛选后: {len(final_cols)}")
    print(f"被剔除的高相关特征: {sorted(list(to_drop))}")
    return final_cols, sorted(list(to_drop))


def fill_missing(feature_df: pd.DataFrame, numeric_cols: list[str], binary_cols: list[str]) -> pd.DataFrame:
    """若存在缺失，再次填补：连续→中位数，二元→众数。"""
    miss = feature_df.isna().sum()
    print("缺失检查：")
    print(miss)
    if miss.sum() == 0:
        print("无缺失。")
        return feature_df
    for col in feature_df.columns:
        if miss[col] == 0:
            continue
        if col in numeric_cols:
            med = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(med)
        elif col in binary_cols:
            mode_series = feature_df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else 0
            feature_df[col] = feature_df[col].fillna(fill_val)
        else:
            mode_series = feature_df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else 0
            feature_df[col] = feature_df[col].fillna(fill_val)
        print(f"{col} 填补完成，缺失数 -> {feature_df[col].isna().sum()}")
    if feature_df.isna().any().any():
        raise ValueError("填补后仍有缺失。")
    return feature_df


def standardize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """连续特征做 Z-score 标准化，0/1 特征保持原样。"""
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            raise ValueError(f"{col} 标准差为 0，无法标准化。")
        df[col] = (df[col] - mean) / std
    return df


def evaluate_kmeans(X: pd.DataFrame, range_k=range(2, 9), random_state=42):
    """多 K 评估，返回结果表。"""
    results = []
    for k in range_k:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        inertia = model.inertia_
        sil = silhouette_score(X, labels)
        results.append({"K": k, "inertia": inertia, "silhouette": sil})
        print(f"K={k}: inertia={inertia:.2f}, silhouette={sil:.4f}")
    return pd.DataFrame(results)


def plot_metric(df_res: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(df_res[x_col], df_res[y_col], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"已保存图表 {path}")


def plot_cluster_means(mean_df: pd.DataFrame, title: str, path: Path):
    """按类别绘制各簇的特征均值条形图。"""
    ax = mean_df.T.plot(kind="bar", figsize=(8, 4))
    ax.set_title(title)
    ax.set_ylabel("Mean (numeric=z-score, binary=proportion)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"已保存特征均值图 {path}")


def map_yes_no(series: pd.Series) -> pd.Series:
    """是/否映射为 1/0，其余缺失。"""
    s = series.map(lambda x: np.nan if pd.isna(x) else str(x).strip())
    yes = {"1", "1.0", "是"}
    no = {"0", "0.0", "否", "2", "2.0"}
    return s.map(lambda x: 1 if x in yes else (0 if x in no else np.nan))


def main():
    df = pd.read_csv(DATA_PATH, encoding=READ_ENCODING)
    if MERGE_KEY not in df.columns:
        raise KeyError(f"{MERGE_KEY} 不在数据中。")

    print(f"读取数据形状: {df.shape}")
    id_series = df[MERGE_KEY]
    feature_df = df.drop(columns=[MERGE_KEY])
    print(f"特征列 ({len(feature_df.columns)}): {list(feature_df.columns)}")
    print("缺失计数（预期为 0）:")
    print(feature_df.isna().sum())

    # 自动识别连续与二元特征
    numeric_features, binary_features = detect_binary_and_numeric(feature_df)

    # 高相关剔除
    numeric_features_final, dropped_for_corr = drop_high_corr(numeric_features, feature_df)

    # 组合用于聚类的特征
    selected_cols = numeric_features_final + binary_features
    feats_for_kmeans = feature_df[selected_cols].copy()

    # 缺失检查与填补（防御性）
    feats_for_kmeans = fill_missing(feats_for_kmeans, numeric_features_final, binary_features)

    # 标准化连续特征（0/1 不标准化）
    feats_for_kmeans = standardize_numeric(feats_for_kmeans, numeric_features_final)

    # 组装最终矩阵
    df_for_kmeans = pd.concat([id_series, feats_for_kmeans], axis=1)
    print(f"df_for_kmeans 形状: {df_for_kmeans.shape}")
    for col in ["land_area", "chem_fert_kg_per_mu", "organic_fert_total", "mobile_hours_per_day"]:
        if col in feats_for_kmeans.columns:
            print(
                f"{col} 标准化后均值 {feats_for_kmeans[col].mean():.4f}, "
                f"标准差 {feats_for_kmeans[col].std():.4f}"
            )
    if df_for_kmeans.drop(columns=[MERGE_KEY]).isna().any().any():
        raise ValueError("标准化后仍存在缺失。")
    if not all(np.issubdtype(dtype, np.number) for dtype in df_for_kmeans.drop(columns=[MERGE_KEY]).dtypes):
        raise TypeError("存在非数值特征，检查数据类型。")

    df_for_kmeans.to_csv(OUTPUT_FEATURES, index=False, encoding=WRITE_ENCODING)
    print(f"已保存用于 K-means 的特征矩阵: {OUTPUT_FEATURES}")

    # 多 K 评估
    X = df_for_kmeans.drop(columns=[MERGE_KEY])
    eval_df = evaluate_kmeans(X, range_k=range(2, 9))
    eval_df.to_csv(OUTPUT_EVAL, index=False, encoding=WRITE_ENCODING)
    plot_metric(eval_df, "K", "inertia", "K vs Inertia (Elbow)", OUTPUT_ELBOW)
    plot_metric(eval_df, "K", "silhouette", "K vs Silhouette", OUTPUT_SIL)

    # 选择 best_k（可按图形人工调整）
    best_k = 3  # 如需调整，请根据 elbow/silhouette 图修改此值
    best_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = best_model.fit_predict(X)

    df_cluster_labels = pd.DataFrame(
        {
            MERGE_KEY: id_series,
            "cluster": clusters,  # 0-based
            "cluster_label": clusters + 1,  # 1-based 便于阅读
        }
    )
    print("各簇样本数:")
    print(df_cluster_labels["cluster"].value_counts().sort_index())

    # 简要查看各簇连续变量均值
    tmp = df_for_kmeans.copy()
    tmp["cluster"] = clusters
    print("按簇的连续特征均值预览:")
    print(tmp.groupby("cluster")[numeric_features_final].mean())

    df_cluster_labels.to_csv(OUTPUT_LABELS, index=False, encoding=WRITE_ENCODING)
    print(f"已保存聚类标签: {OUTPUT_LABELS}（cluster 从 0 开始，cluster_label 为 1..{best_k}）")

    # 额外：按主题分组的特征均值图（使用聚类输入的尺度：连续为Z-score，二元为比例）
    cluster_means = tmp.groupby("cluster")[selected_cols].mean()
    resource_cols = [
        "land_area",
        "plot_num",
        "tillage_mech_ratio",
        "chem_fert_kg_per_mu",
        "organic_fert_buy",
        "irrigated_area_ratio",
        "organic_fert_total",
    ]
    digital_cols = [
        "mobile_hours_per_day",
        "can_get_info_online",
        "internet_training",
        "e_commerce",
    ]
    financial_cols = [
        "ag_insurance",
        "bank_credit_user",
        "applied_bank_loan",
        "trade_credit",
    ]
    health_env_cols = [
        "safe_drinking_water",
        "garbage_sorting",
        "garbage_central_treatment",
        "sanitary_toilet",
        "health_knowledge_learning",
        "has_checkup",
        "straw_utilized",
        "pesticide_pack_safe",
        "diet_control_score",
    ]
    plot_cluster_means(cluster_means[resource_cols], "资源禀赋/生产基础 - 簇均值", OUTPUT_MEAN_RES)
    plot_cluster_means(cluster_means[digital_cols], "数字与信息能力 - 簇均值", OUTPUT_MEAN_DIG)
    plot_cluster_means(cluster_means[financial_cols], "金融行为 - 簇均值", OUTPUT_MEAN_FIN)
    plot_cluster_means(cluster_means[health_env_cols], "健康与环境 - 簇均值", OUTPUT_MEAN_HE)

    # 额外：cluster × 合作社成员交叉表 + 收入/满意度均值
    raw_df = pd.read_csv(RAW_PATH, encoding=READ_ENCODING)
    coop = map_yes_no(raw_df.get("var9_17", pd.Series([np.nan] * len(raw_df))))
    income = pd.to_numeric(raw_df.get("var17_1", pd.Series([np.nan] * len(raw_df))), errors="coerce")
    # 假设 var20_7 为满意度/幸福感得分（如有不同请替换变量名）
    satisfaction = pd.to_numeric(raw_df.get("var20_7", pd.Series([np.nan] * len(raw_df))), errors="coerce")
    aux = pd.DataFrame(
        {MERGE_KEY: raw_df[MERGE_KEY], "coop_member": coop, "income": income, "satisfaction": satisfaction}
    )
    merged = df_cluster_labels.merge(aux, on=MERGE_KEY, how="left")
    crosstab = pd.crosstab(merged["cluster"], merged["coop_member"])
    print("cluster × 合作社成员（1=是，0=否）:")
    print(crosstab)
    crosstab.to_csv(OUTPUT_CROSSTAB, encoding=WRITE_ENCODING)

    income_satis = merged.groupby("cluster")[["income", "satisfaction"]].mean()
    print("cluster 收入/满意度均值:")
    print(income_satis)
    income_satis.to_csv(OUTPUT_CLUSTER_SUMMARY, encoding=WRITE_ENCODING)


if __name__ == "__main__":
    main()
