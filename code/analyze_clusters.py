import numpy as np
import pandas as pd
from pathlib import Path

# 路径定义
BASE = Path(__file__).resolve().parent
CLUSTER_LABELS_PATH = BASE / "cluster_with_labels.csv"  # 使用 k=3 结果
CLUSTER_DF_PATH = BASE / "cluster_df.csv"
DATA_DIR = BASE.parent / "农户"
HH5_NAME = "hh_5.csv"
HH17_NAME = "hh_17.csv"
HH11_NAME = "hh_11.csv"

# 输出路径
OUT_DF_ALL = BASE / "df_all_with_new_mcda.csv"
OUT_BEHAVIOR = BASE / "cluster_behavior_profile.csv"
OUT_MEMBER_CROSSTAB = BASE / "cluster_member_crosstab.csv"
OUT_OUTCOME = BASE / "cluster_outcome_summary.csv"
OUT_MCDA_MEAN = BASE / "cluster_mcda_mean.csv"
OUT_MEMBER_MCDA = BASE / "cluster_member_mcda_mean.csv"
OUT_MEMBER_GAP = BASE / "cluster_member_gap_mcda.csv"

MERGE_KEY = "varID"
SPECIAL_MISSING = {".", "", "8888", "8888.0", "9999", "9999.0", "不知道", 8888, 9999}


def read_dedup(path: Path, key: str) -> pd.DataFrame:
    """读取并确保 key 唯一，若重复保留首行。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if key not in df.columns:
        raise KeyError(f"{key} 不在文件 {path}")
    dup_count = df.duplicated(subset=key).sum()
    if dup_count:
        print(f"{path.name} 存在 {dup_count} 个重复 {key}，保留首行删除其余。")
        df = df.drop_duplicates(subset=key, keep="first")
    return df


def find_data_file(preferred: str, fallback_pattern: str) -> Path:
    """在 DATA_DIR 下查找文件，优先 preferred，否则按 glob fallback_pattern。"""
    pref_path = DATA_DIR / preferred
    if pref_path.exists():
        return pref_path
    matches = list(DATA_DIR.glob(fallback_pattern))
    if not matches:
        raise FileNotFoundError(f"未找到 {preferred} 或 {fallback_pattern}")
    if len(matches) > 1:
        print(f"找到多个匹配 {fallback_pattern}，使用 {matches[0]}")
    return matches[0]


def winsorize(series: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    """按分位数截尾，避免极端值。"""
    if series.dropna().empty:
        return series
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def minmax_scale(series: pd.Series) -> pd.Series:
    """线性缩放到[0,1]，若常数列则返回 0。"""
    s = series
    min_v, max_v = s.min(), s.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_v) / (max_v - min_v)


def encode_yes_no(series: pd.Series) -> pd.Series:
    """是/否编码为 1/0，其他为缺失。"""
    s = series.map(lambda x: np.nan if pd.isna(x) else str(x).strip())
    yes = {"1", "1.0", "是"}
    no = {"0", "0.0", "否", "2", "2.0"}
    return s.map(lambda x: 1 if x in yes else (0 if x in no else np.nan))


def main():
    # Step 1: 读取数据并合并
    cluster_labels = read_dedup(CLUSTER_LABELS_PATH, MERGE_KEY)
    cluster_df = read_dedup(CLUSTER_DF_PATH, MERGE_KEY)
    hh5_path = find_data_file(HH5_NAME, "hh_5_*.csv")
    hh17_path = find_data_file(HH17_NAME, "hh_17*.csv")
    hh11_path = find_data_file(HH11_NAME, "hh_11*.csv")
    hh5 = read_dedup(hh5_path, MERGE_KEY)
    hh17 = read_dedup(hh17_path, MERGE_KEY)
    hh11 = read_dedup(hh11_path, MERGE_KEY)

    # 列名提示（便于修正）
    for name, df in [
        ("hh5", hh5),
        ("hh17", hh17),
        ("hh11", hh11),
    ]:
        print(f"{name} 列名示例: {list(df.columns)[:10]}")

    # 逐步内连接并打印样本数
    # 仅保留聚类标签，避免重复特征
    base_cols = [c for c in ["cluster", "cluster_label"] if c in cluster_labels.columns]
    df_all = cluster_labels[[MERGE_KEY] + base_cols].copy()
    for name, df in [
        ("cluster_df", cluster_df),
        ("hh5", hh5[[MERGE_KEY, "var5_1"]] if "var5_1" in hh5.columns else hh5),
        ("hh17", hh17[[MERGE_KEY, "var17_32_1"]] if "var17_32_1" in hh17.columns else hh17),
        (
            "hh11",
            hh11[
                [MERGE_KEY, "var11_1", "var11_2", "var11_3", "var11_4"]
            ]
            if set(["var11_1", "var11_2", "var11_3", "var11_4"]).issubset(hh11.columns)
            else hh11,
        ),
    ]:
        before = len(df_all)
        df_all = df_all.merge(df, on=MERGE_KEY, how="inner")
        print(f"与 {name} 合并: {before} -> {len(df_all)} 行")

    # 统一替换特殊缺失编码
    df_all = df_all.replace(SPECIAL_MISSING, np.nan)

    print("df_all 形状:", df_all.shape)
    key_cols = [
        "cluster",
        "land_area",
        "chem_fert_kg_per_mu",
        "mobile_hours_per_day",
        "ag_insurance",
        "straw_utilized",
        "pesticide_pack_safe",
        "var5_1",
        "var17_32_1",
        "var11_1",
        "var11_2",
        "var11_3",
        "var11_4",
    ]
    for col in key_cols:
        if col in df_all.columns:
            print(f"{col} 缺失数: {df_all[col].isna().sum()}")
        else:
            print(f"{col} 不在 df_all 中，请检查列名。")

    # var5_1 生成 member
    if "var5_1" in df_all.columns:
        df_all["member"] = encode_yes_no(df_all["var5_1"])
        print("member 取值频数:")
        print(df_all["member"].value_counts(dropna=False))
    else:
        df_all["member"] = np.nan
        print("缺少 var5_1，member 置为缺失。")

    # 将关键结果变量转为数值
    num_cols_convert = ["var17_32_1", "var11_1", "var11_2", "var11_3", "var11_4"]
    for col in num_cols_convert:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    # Step 2: 行为画像 + 交叉 + 结果
    behavior_vars = [
        "land_area",
        "plot_num",
        "tillage_mech_ratio",
        "irrigated_area_ratio",
        "chem_fert_kg_per_mu",
        "organic_fert_total",
        "mobile_hours_per_day",
        "can_get_info_online",
        "internet_training",
        "e_commerce",
        "ag_insurance",
        "bank_credit_user",
        "applied_bank_loan",
        "trade_credit",
        "safe_drinking_water",
        "garbage_sorting",
        "garbage_central_treatment",
        "sanitary_toilet",
        "health_knowledge_learning",
        "has_checkup",
        "straw_utilized",
        "pesticide_pack_safe",
    ]
    use_vars = [c for c in behavior_vars if c in df_all.columns]
    cluster_behavior_profile = df_all.groupby("cluster")[use_vars].mean()
    cluster_behavior_profile.to_csv(OUT_BEHAVIOR, encoding="utf-8-sig")
    print(f"已保存 cluster_behavior_profile -> {OUT_BEHAVIOR}")

    # cluster × member 列联表（计数与行比例）
    crosstab_cnt = pd.crosstab(df_all["cluster"], df_all["member"])
    crosstab_pct = crosstab_cnt.div(crosstab_cnt.sum(axis=1), axis=0)
    crosstab_all = pd.concat({"count": crosstab_cnt, "row_pct": crosstab_pct}, axis=1)
    crosstab_all.to_csv(OUT_MEMBER_CROSSTAB, encoding="utf-8-sig")
    print(f"已保存 cluster_member_crosstab -> {OUT_MEMBER_CROSSTAB}")

    # 收入/满意度结果
    outcome_cols = ["var17_32_1", "var11_1", "var11_2", "var11_3", "var11_4"]
    outcome_exist = [c for c in outcome_cols if c in df_all.columns]
    stats = {}
    if "var17_32_1" in outcome_exist:
        stats["income_mean"] = df_all.groupby("cluster")["var17_32_1"].mean()
        stats["income_median"] = df_all.groupby("cluster")["var17_32_1"].median()
        stats["income_std"] = df_all.groupby("cluster")["var17_32_1"].std()
    for col in ["var11_1", "var11_2", "var11_3", "var11_4"]:
        if col in outcome_exist:
            stats[f"{col}_mean"] = df_all.groupby("cluster")[col].mean()
    cluster_outcome_summary = pd.DataFrame(stats)
    cluster_outcome_summary.to_csv(OUT_OUTCOME, encoding="utf-8-sig")
    print(f"已保存 cluster_outcome_summary -> {OUT_OUTCOME}")

    # Step 3: 新 MCDA 指数
    # 3.1 CMI
    cmi_cont = ["land_area", "tillage_mech_ratio", "irrigated_area_ratio", "mobile_hours_per_day"]
    cmi_bin = [
        "can_get_info_online",
        "internet_training",
        "e_commerce",
        "ag_insurance",
        "bank_credit_user",
        "applied_bank_loan",
        "trade_credit",
    ]
    df_all_cmi = df_all.copy()
    # 连续：填补中位数 -> winsor -> minmax
    for col in cmi_cont:
        if col not in df_all_cmi.columns:
            df_all_cmi[col] = np.nan
        df_all_cmi[col] = pd.to_numeric(df_all_cmi[col], errors="coerce")
        med = df_all_cmi[col].median()
        if pd.isna(med):
            med = 0
        df_all_cmi[col] = df_all_cmi[col].fillna(med)
        df_all_cmi[col] = winsorize(df_all_cmi[col])
        df_all_cmi[col] = minmax_scale(df_all_cmi[col])
    # 二元：众数填补
    for col in cmi_bin:
        if col not in df_all_cmi.columns:
            df_all_cmi[col] = np.nan
        df_all_cmi[col] = pd.to_numeric(df_all_cmi[col], errors="coerce")
        mode_series = df_all_cmi[col].mode(dropna=True)
        fill_val = mode_series.iloc[0] if not mode_series.empty else 0
        df_all_cmi[col] = df_all_cmi[col].fillna(fill_val)
    df_all["CMI"] = df_all_cmi[cmi_cont + cmi_bin].mean(axis=1)
    print("CMI 描述统计:")
    print(df_all["CMI"].describe())

    # 3.2 GSPI
    gspi_cont = ["chem_fert_kg_per_mu", "organic_fert_total"]
    gspi_bin = ["straw_utilized", "pesticide_pack_safe", "garbage_sorting", "garbage_central_treatment"]
    df_all_gspi = df_all.copy()
    for col in gspi_cont:
        if col not in df_all_gspi.columns:
            df_all_gspi[col] = np.nan
        df_all_gspi[col] = pd.to_numeric(df_all_gspi[col], errors="coerce")
        med = df_all_gspi[col].median()
        if pd.isna(med):
            med = 0
        df_all_gspi[col] = df_all_gspi[col].fillna(med)
        df_all_gspi[col] = winsorize(df_all_gspi[col])
        df_all_gspi[col] = minmax_scale(df_all_gspi[col])
    # 反向化肥
    df_all_gspi["chem_fert_green"] = 1 - df_all_gspi["chem_fert_kg_per_mu"]
    df_all_gspi["organic_fert_score"] = df_all_gspi["organic_fert_total"]
    for col in gspi_bin:
        if col not in df_all_gspi.columns:
            df_all_gspi[col] = np.nan
        df_all_gspi[col] = pd.to_numeric(df_all_gspi[col], errors="coerce")
        mode_series = df_all_gspi[col].mode(dropna=True)
        fill_val = mode_series.iloc[0] if not mode_series.empty else 0
        df_all_gspi[col] = df_all_gspi[col].fillna(fill_val)
    gspi_cols = ["chem_fert_green", "organic_fert_score"] + gspi_bin
    df_all["GSPI"] = df_all_gspi[gspi_cols].mean(axis=1)
    print("GSPI 描述统计:")
    print(df_all["GSPI"].describe())

    # 3.3 HLEI
    hlei_bin = ["safe_drinking_water", "sanitary_toilet", "health_knowledge_learning", "has_checkup"]
    df_all_hlei = df_all.copy()
    for col in hlei_bin:
        if col not in df_all_hlei.columns:
            df_all_hlei[col] = np.nan
        df_all_hlei[col] = pd.to_numeric(df_all_hlei[col], errors="coerce")
        mode_series = df_all_hlei[col].mode(dropna=True)
        fill_val = mode_series.iloc[0] if not mode_series.empty else 0
        df_all_hlei[col] = df_all_hlei[col].fillna(fill_val)
    df_all["HLEI"] = df_all_hlei[hlei_bin].mean(axis=1)
    print("HLEI 描述统计:")
    print(df_all["HLEI"].describe())

    # 3.4 总体指数
    df_all["Overall_new"] = df_all[["CMI", "GSPI", "HLEI"]].mean(axis=1)
    print("Overall_new 描述统计:")
    print(df_all["Overall_new"].describe())

    # 保存含新指数的总表
    df_all.to_csv(OUT_DF_ALL, index=False, encoding="utf-8-sig")
    print(f"已保存 df_all_with_new_mcda -> {OUT_DF_ALL}")

    # Step 4: MCDA 按 cluster / member 分析
    mcda_cols = ["CMI", "GSPI", "HLEI", "Overall_new"]
    cluster_mcda_mean = df_all.groupby("cluster")[mcda_cols].agg(["mean", "std"])
    cluster_mcda_mean.to_csv(OUT_MCDA_MEAN, encoding="utf-8-sig")
    print(f"已保存 cluster_mcda_mean -> {OUT_MCDA_MEAN}")

    cluster_member_mcda_mean = (
        df_all.groupby(["cluster", "member"])[mcda_cols].mean().reset_index()
    )
    cluster_member_mcda_mean.to_csv(OUT_MEMBER_MCDA, index=False, encoding="utf-8-sig")
    print(f"已保存 cluster_member_mcda_mean -> {OUT_MEMBER_MCDA}")

    # 成员-非成员差值
    gaps = []
    for clus, sub in df_all.groupby("cluster"):
        member_mean = sub[sub["member"] == 1][mcda_cols].mean()
        nonmember_mean = sub[sub["member"] == 0][mcda_cols].mean()
        gap = member_mean - nonmember_mean
        gap["cluster"] = clus
        gaps.append(gap)
    cluster_member_gap = pd.DataFrame(gaps).set_index("cluster")
    cluster_member_gap.to_csv(OUT_MEMBER_GAP, encoding="utf-8-sig")
    print(f"已保存 cluster_member_gap_mcda -> {OUT_MEMBER_GAP}")

    # 末尾打印关键信息
    print("每个 cluster 的样本数与成员比例:")
    print(df_all.groupby("cluster")["member"].agg(["count", "mean"]))
    print("每个 cluster 的 MCDA 均值:")
    print(df_all.groupby("cluster")[mcda_cols].mean())
    print("Overall_new：成员 vs 非成员差值（按 cluster）:")
    for clus, sub in df_all.groupby("cluster"):
        member_mean = sub[sub["member"] == 1]["Overall_new"].mean()
        nonmember_mean = sub[sub["member"] == 0]["Overall_new"].mean()
        print(f"cluster {clus}: member - nonmember = {member_mean - nonmember_mean:.4f}")


if __name__ == "__main__":
    main()
